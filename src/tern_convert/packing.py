"""Ternary (tern5) packing and unpacking utilities.

Packing: 5 trits per byte using base-3 encoding.
- Each trit in {-1, 0, 1} is encoded as digit = trit + 1 in {0, 1, 2}
- Byte value = sum(digit_i * 3^i) for i=0..4, range 0..242
- Each row is packed independently, padded with zeros to multiple of 5
- bytes_per_row = ceil(row_length / 5)
"""

from __future__ import annotations

import numpy as np

# Constants
TRITS_PER_BYTE = 5
MAX_PACKED_VALUE = 3**TRITS_PER_BYTE - 1  # 242
BASE = 3

# Precomputed powers of 3
POW3 = np.array([BASE**i for i in range(TRITS_PER_BYTE)], dtype=np.uint8)


def pack_ternary_row(trits: np.ndarray) -> np.ndarray:
    """Pack a 1D array of trits (int8 in {-1,0,1}) into bytes (uint8).

    Args:
        trits: 1D array of trits, values in {-1, 0, 1}

    Returns:
        1D array of packed bytes (uint8), each byte holds 5 trits
    """
    if trits.dtype != np.int8:
        trits = trits.astype(np.int8)

    n = len(trits)
    # Pad to multiple of 5
    pad_len = (-n) % TRITS_PER_BYTE
    if pad_len:
        trits = np.pad(trits, (0, pad_len), mode='constant', constant_values=0)

    n_padded = len(trits)
    n_bytes = n_padded // TRITS_PER_BYTE

    # Reshape to (n_bytes, 5)
    trits_reshaped = trits.reshape(n_bytes, TRITS_PER_BYTE)

    # Convert trits to digits: trit + 1 -> {0, 1, 2}
    digits = trits_reshaped + 1  # Still int8

    # Compute byte values: sum(digit_i * 3^i)
    # Use broadcasting: digits * POW3 -> sum along axis 1
    packed = np.sum(digits * POW3, axis=1, dtype=np.uint16).astype(np.uint8)

    return packed


def unpack_ternary_row(packed: np.ndarray, n_trits: int) -> np.ndarray:
    """Unpack bytes back to trits.

    Args:
        packed: 1D array of packed bytes (uint8)
        n_trits: Original number of trits (before padding)

    Returns:
        1D array of trits (int8 in {-1, 0, 1}) of length n_trits
    """
    n_bytes = len(packed)
    # Each byte gives 5 trits
    trits_padded = np.zeros(n_bytes * TRITS_PER_BYTE, dtype=np.int8)

    for i in range(n_bytes):
        # Use .item() to get a Python int, not a numpy scalar view
        # This prevents modifying the original packed array
        byte_val = int(packed[i])
        # Extract 5 base-3 digits
        for j in range(TRITS_PER_BYTE):
            digit = byte_val % BASE
            byte_val //= BASE
            # Convert digit back to trit: digit - 1 -> {-1, 0, 1}
            trits_padded[i * TRITS_PER_BYTE + j] = digit - 1

    # Trim padding
    return trits_padded[:n_trits]


def pack_ternary_matrix(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack a 2D matrix (out_features, in_features) row by row.

    Args:
        matrix: 2D array of shape (out, in), values in {-1, 0, 1}

    Returns:
        (packed_bytes, bytes_per_row)
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

    out_features, in_features = matrix.shape
    bytes_per_row = (in_features + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
    total_bytes = out_features * bytes_per_row

    packed = np.zeros(total_bytes, dtype=np.uint8)

    for i in range(out_features):
        row = matrix[i]
        row_packed = pack_ternary_row(row)
        start = i * bytes_per_row
        packed[start:start + bytes_per_row] = row_packed

    return packed, bytes_per_row


def unpack_ternary_matrix(packed: np.ndarray, shape: tuple[int, int], bytes_per_row: int) -> np.ndarray:
    """Unpack bytes back to 2D matrix.

    Args:
        packed: 1D array of packed bytes
        shape: (out_features, in_features)
        bytes_per_row: Number of bytes per row (from pack_ternary_matrix)

    Returns:
        2D array of trits (int8) of given shape
    """
    out_features, in_features = shape
    matrix = np.zeros(shape, dtype=np.int8)

    for i in range(out_features):
        start = i * bytes_per_row
        row_packed = packed[start:start + bytes_per_row]
        matrix[i] = unpack_ternary_row(row_packed, in_features)

    return matrix


def validate_packed_bytes(packed: np.ndarray) -> bool:
    """Check that all packed bytes are valid (<= 242)."""
    return np.all(packed <= MAX_PACKED_VALUE)


def validate_trits(trits: np.ndarray) -> bool:
    """Check that all values are valid trits (-1, 0, 1)."""
    return np.all((trits >= -1) & (trits <= 1))


# Vectorized unpack for performance
def unpack_ternary_row_vec(packed: np.ndarray, n_trits: int) -> np.ndarray:
    """Vectorized unpack of multiple rows at once."""
    n_bytes = len(packed)
    # Use broadcasting to extract all digits at once
    # Shape: (n_bytes, 5)
    bytes_2d = packed[:, None]  # (n_bytes, 1)
    pow3_2d = POW3[None, :]     # (1, 5)

    # For each byte, we need to extract base-3 digits
    # This is harder to fully vectorize, so we use a loop over bytes
    # but vectorize the digit extraction per byte
    trits_padded = np.zeros(n_bytes * TRITS_PER_BYTE, dtype=np.int8)

    for i in range(n_bytes):
        byte_val = int(packed[i])
        # Extract 5 base-3 digits
        for j in range(TRITS_PER_BYTE):
            digit = byte_val % BASE
            byte_val //= BASE
            trits_padded[i * TRITS_PER_BYTE + j] = digit - 1

    return trits_padded[:n_trits]