"""Quantization logic for ternary conversion.

Two modes:
1. exact: verify weights are already ternary (within tolerance)
2. quantize: apply absmean quantization (BitNet b1.58 style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .packing import pack_ternary_matrix, TRITS_PER_BYTE


@dataclass
class QuantizeResult:
    """Result of quantization for a single tensor."""
    name: str
    shape: tuple[int, ...]
    original_dtype: str
    group_size: int
    # For exact mode
    is_ternary: bool | None = None
    max_deviation: float | None = None
    # For quantize mode
    packed: np.ndarray | None = None
    scales: np.ndarray | None = None
    bytes_per_row: int | None = None
    rel_error: float | None = None
    zero_fraction: float | None = None


def check_exact_ternary(
    weight: np.ndarray,
    group_size: int,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Check if a weight matrix is already ternary.

    Args:
        weight: 2D weight matrix (out_features, in_features)
        group_size: Group size for scaling. 0 = per-tensor (one scale for all weights)
        tolerance: Maximum allowed deviation from ternary values

    Returns:
        (is_ternary, max_deviation)
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {weight.ndim}D")

    out_features, in_features = weight.shape

    if group_size == 0:
        # One scale per tensor - check the whole matrix with one scale
        scale = np.max(np.abs(weight))
        if scale < tolerance:
            # All zeros - trivially ternary with scale 1
            return True, 0.0

        normalized = weight / scale
        rounded = np.round(normalized)
        rounded = np.clip(rounded, -1, 1)

        deviation = np.abs(normalized - rounded)
        max_dev = np.max(deviation)

        return max_dev <= tolerance, max_dev
    else:
        # Groups along input dimension, one scale per group per row
        n_groups = (in_features + group_size - 1) // group_size
        max_dev = 0.0

        for i in range(out_features):
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, in_features)
                group_weights = weight[i, start:end]

                if len(group_weights) == 0:
                    continue

                scale = np.max(np.abs(group_weights))
                if scale < tolerance:
                    continue

                normalized = group_weights / scale
                rounded = np.clip(np.round(normalized), -1, 1)

                group_max_dev = np.max(np.abs(normalized - rounded))
                max_dev = max(max_dev, group_max_dev)

                if group_max_dev > tolerance:
                    return False, max_dev

        return True, max_dev


def quantize_absmean(
    weight: np.ndarray,
    group_size: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply absmean quantization (BitNet b1.58 style).

    Args:
        weight: 2D weight matrix (out_features, in_features)
        group_size: Group size for scaling. 0 = per-tensor (one scale for entire tensor)
        eps: Small epsilon for numerical stability

    Returns:
        (packed_bytes, scales, bytes_per_row)
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {weight.ndim}D")

    out_features, in_features = weight.shape

    if group_size == 0:
        # One scale per tensor
        scale = np.mean(np.abs(weight)) + eps
        scales = np.array([scale], dtype=np.float32)

        if scale > 0:
            quantized = np.round(weight / scale)
            quantized = np.clip(quantized, -1, 1)
            trits_matrix = quantized.astype(np.int8)
        else:
            trits_matrix = np.zeros_like(weight, dtype=np.int8)

    else:
        # Groups along input dimension, one scale per group per row
        n_groups = (in_features + group_size - 1) // group_size
        scales = np.zeros((out_features, n_groups), dtype=np.float32)
        trits_matrix = np.zeros_like(weight, dtype=np.int8)

        for i in range(out_features):
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, in_features)
                group = weight[i, start:end]

                scale = np.mean(np.abs(group)) + eps
                scales[i, g] = scale

                if scale > 0:
                    quantized = np.round(group / scale)
                    quantized = np.clip(quantized, -1, 1)
                    trits_matrix[i, start:end] = quantized.astype(np.int8)
                else:
                    trits_matrix[i, start:end] = 0

        # Flatten scales for storage
        scales = scales.flatten()

    # Pack trits
    packed, bytes_per_row = pack_ternary_matrix(trits_matrix)

    return packed, scales, bytes_per_row


def compute_quantization_error(
    original: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bytes_per_row: int,
) -> tuple[float, float]:
    """Compute relative error and zero fraction after quantization.

    Args:
        original: Original weight matrix
        packed: Packed ternary bytes
        scales: Scale values (1 element if group_size==0, else out_features * n_groups)
        group_size: Group size (0 = per-tensor)
        bytes_per_row: Bytes per row from packing

    Returns:
        (relative_error, zero_fraction)
    """
    from .packing import unpack_ternary_matrix

    # Unpack
    trits = unpack_ternary_matrix(packed, original.shape, bytes_per_row)

    # Reconstruct
    out_features, in_features = original.shape

    if group_size == 0:
        # One scale per tensor
        scale = scales[0]  # Single scale for entire tensor
        reconstructed = trits.astype(np.float32) * scale
    else:
        n_groups = (in_features + group_size - 1) // group_size
        scales_reshaped = scales.reshape(out_features, n_groups)
        reconstructed = np.zeros_like(trits, dtype=np.float32)
        for i in range(out_features):
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, in_features)
                reconstructed[i, start:end] = trits[i, start:end] * scales_reshaped[i, g]

    # Relative Frobenius error
    diff = reconstructed - original.astype(np.float32)
    rel_error = np.linalg.norm(diff, 'fro') / (np.linalg.norm(original, 'fro') + 1e-12)

    # Zero fraction
    zero_frac = np.mean(trits == 0)

    return float(rel_error), float(zero_frac)


def process_tensor(
    name: str,
    weight: np.ndarray,
    mode: Literal["exact", "quantize"],
    group_size: int,
    tolerance: float = 1e-6,
    eps: float = 1e-8,
) -> QuantizeResult:
    """Process a single tensor according to mode.

    Args:
        name: Tensor name
        weight: Weight matrix (2D)
        mode: "exact" or "quantize"
        group_size: Group size for scaling
        tolerance: Tolerance for exact mode
        eps: Epsilon for quantize mode

    Returns:
        QuantizeResult with all relevant info
    """
    original_dtype = str(weight.dtype)

    if mode == "exact":
        is_ternary, max_dev = check_exact_ternary(weight, group_size, tolerance)
        return QuantizeResult(
            name=name,
            shape=weight.shape,
            original_dtype=original_dtype,
            group_size=group_size,
            is_ternary=is_ternary,
            max_deviation=max_dev,
        )
    else:
        packed, scales, bytes_per_row = quantize_absmean(weight, group_size, eps)
        rel_error, zero_frac = compute_quantization_error(
            weight, packed, scales, group_size, bytes_per_row
        )
        return QuantizeResult(
            name=name,
            shape=weight.shape,
            original_dtype=original_dtype,
            group_size=group_size,
            packed=packed,
            scales=scales,
            bytes_per_row=bytes_per_row,
            rel_error=rel_error,
            zero_fraction=zero_frac,
        )