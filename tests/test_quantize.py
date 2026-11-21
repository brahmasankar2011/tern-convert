"""Tests for quantization logic."""

import numpy as np
import pytest

from src.tern_convert.quantize import (
    check_exact_ternary,
    quantize_absmean,
    compute_quantization_error,
    process_tensor,
    QuantizeResult,
)


class TestCheckExactTernary:
    def test_perfect_ternary_per_row(self):
        """Test matrix with exact ternary values per row."""
        weight = np.array([
            [1.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, -1.0, 0.0],
        ], dtype=np.float32)

        is_ternary, max_dev = check_exact_ternary(weight, group_size=0)
        assert is_ternary
        assert max_dev < 1e-6

    def test_perfect_ternary_with_groups(self):
        """Test matrix with exact ternary values with groups."""
        weight = np.array([
            [1.0, 0.0, -1.0, 1.0, 0.0, -1.0],
            [0.5, 0.0, -0.5, 0.5, 0.0, -0.5],  # scale=0.5
        ], dtype=np.float32)

        is_ternary, max_dev = check_exact_ternary(weight, group_size=3)
        assert is_ternary
        assert max_dev < 1e-6

    def test_near_ternary_rejected(self):
        """Test that near-ternary values are rejected in exact mode."""
        weight = np.array([
            [1.0, 0.0, -1.0, 1.0],
            [0.99, 0.0, -1.01, 0.0],  # Slightly off
        ], dtype=np.float32)

        is_ternary, max_dev = check_exact_ternary(weight, group_size=0, tolerance=1e-3)
        assert not is_ternary
        assert max_dev > 1e-3

    def test_near_ternary_accepted_with_loose_tolerance(self):
        """Test that near-ternary values pass with loose tolerance."""
        weight = np.array([
            [1.0, 0.0, -1.0, 1.0],
            [0.99, 0.0, -1.01, 0.0],
        ], dtype=np.float32)

        is_ternary, max_dev = check_exact_ternary(weight, group_size=0, tolerance=0.02)
        assert is_ternary
        assert max_dev < 0.02

    def test_all_zeros(self):
        """Test all-zero matrix."""
        weight = np.zeros((3, 4), dtype=np.float32)
        is_ternary, max_dev = check_exact_ternary(weight, group_size=0)
        assert is_ternary
        assert max_dev == 0.0

    def test_3d_rejected(self):
        """Test that non-2D arrays are rejected."""
        weight = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            check_exact_ternary(weight, group_size=0)

    def test_per_tensor_scale_used_for_entire_matrix(self):
        """Test that group_size=0 uses one scale for entire matrix.

        For per-tensor mode, the scale is the max absolute value in the entire matrix,
        and all values must be close to {-scale, 0, scale}.
        """
        # Each row has different magnitude, but all are ternary with same scale
        # Using scale=2, values are {-2, 0, 2}
        weight = np.array([
            [2.0, 0.0, -2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [-2.0, 2.0, -2.0, 0.0],
        ], dtype=np.float32)

        is_ternary, max_dev = check_exact_ternary(weight, group_size=0)
        assert is_ternary
        assert max_dev < 1e-6

    def test_per_tensor_scale_rejects_row_different_scales(self):
        """Test that per-tensor mode rejects when rows need different scales.

        If each row would need its own scale for ternary representation,
        per-tensor mode should still accept it (using the overall max scale).
        """
        # These values CAN be represented with a single scale=2
        weight = np.array([
            [2.0, 0.0, -2.0, 1.0],  # 1.0 is not {-2, 0, 2}
        ], dtype=np.float32)

        is_ternary, max_dev = check_exact_ternary(weight, group_size=0, tolerance=0.1)
        # 1.0/2 = 0.5, rounds to 1, clipped to 1, gives scale*1=2, not 1
        # So max_dev should be |0.5 - 1| = 0.5
        assert not is_ternary  # Cannot represent with per-tensor scale
        assert max_dev > 0.1


class TestQuantizeAbsmean:
    def test_basic_quantization(self):
        """Test basic absmean quantization."""
        weight = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, -2.0, -3.0, -4.0],
        ], dtype=np.float32)

        packed, scales, bytes_per_row = quantize_absmean(weight, group_size=0)

        # group_size=0 means one scale per tensor
        assert scales.shape == (1,)
        assert packed.dtype == np.uint8
        assert bytes_per_row == 1  # 4 inputs -> ceil(4/5) = 1

    def test_quantization_with_groups(self):
        """Test quantization with group_size > 0."""
        weight = np.arange(24, dtype=np.float32).reshape(3, 8) - 12

        packed, scales, bytes_per_row = quantize_absmean(weight, group_size=4)

        # 3 rows, 2 groups per row (8/4)
        assert scales.shape == (3 * 2,)
        assert bytes_per_row == 2  # 8 inputs -> ceil(8/5) = 2

    def test_quantization_error_computation(self):
        """Test that error computation works."""
        weight = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ], dtype=np.float32)

        packed, scales, bytes_per_row = quantize_absmean(weight, group_size=0)
        rel_error, zero_frac = compute_quantization_error(
            weight, packed, scales, group_size=0, bytes_per_row=bytes_per_row
        )

        assert 0 <= rel_error < 1.0
        assert 0 <= zero_frac <= 1.0

    def test_quantization_error_with_groups(self):
        """Test error computation with groups."""
        weight = np.arange(24, dtype=np.float32).reshape(3, 8) - 12

        packed, scales, bytes_per_row = quantize_absmean(weight, group_size=4)
        rel_error, zero_frac = compute_quantization_error(
            weight, packed, scales, group_size=4, bytes_per_row=bytes_per_row
        )

        assert 0 <= rel_error < 1.0
        assert 0 <= zero_frac <= 1.0

    def test_eps_prevents_division_by_zero(self):
        """Test that eps prevents issues with all-zero tensor."""
        weight = np.zeros((2, 4), dtype=np.float32)

        packed, scales, bytes_per_row = quantize_absmean(weight, group_size=0, eps=1e-8)
        # All-Zero tensor gets single scale for entire tensor
        assert scales.shape == (1,)
        assert scales[0] == 1e-8


class TestProcessTensor:
    def test_process_exact_mode_ternary(self):
        """Test processing a ternary tensor in exact mode."""
        weight = np.array([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        result = process_tensor("test", weight, mode="exact", group_size=0)

        assert isinstance(result, QuantizeResult)
        assert result.name == "test"
        assert result.shape == (2, 3)
        assert result.original_dtype == "float32"
        assert result.group_size == 0
        assert result.is_ternary == True
        assert result.max_deviation is not None
        assert result.max_deviation < 1e-6

    def test_process_exact_mode_not_ternary(self):
        """Test processing a non-ternary tensor in exact mode."""
        weight = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ], dtype=np.float32)

        result = process_tensor("test", weight, mode="exact", group_size=0)

        assert result.is_ternary == False
        assert result.max_deviation > 1e-6

    def test_process_quantize_mode(self):
        """Test processing a tensor in quantize mode."""
        weight = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ], dtype=np.float32)

        result = process_tensor("test", weight, mode="quantize", group_size=0)

        assert result.is_ternary is None  # Not set in quantize mode
        assert result.packed is not None
        assert result.scales is not None
        assert result.bytes_per_row is not None
        assert result.rel_error is not None
        assert 0 <= result.rel_error < 1.0
        assert result.zero_fraction is not None
        assert 0 <= result.zero_fraction <= 1.0

    def test_process_quantize_mode_with_groups(self):
        """Test quantize mode with group_size > 0."""
        weight = np.arange(24, dtype=np.float32).reshape(3, 8) - 12

        result = process_tensor("test", weight, mode="quantize", group_size=4)

        assert result.group_size == 4
        assert result.scales.shape == (6,)  # 3 rows * 2 groups
        assert result.rel_error is not None

    def test_process_quantize_mode_per_tensor(self):
        """Test quantize mode with group_size=0 (per-tensor, one scale for entire matrix)."""
        weight = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ], dtype=np.float32)

        result = process_tensor("test", weight, mode="quantize", group_size=0)

        assert result.group_size == 0
        # One scale for entire tensor
        assert result.scales.shape == (1,)
        assert result.packed is not None
        assert result.bytes_per_row is not None
        assert result.rel_error is not None
        assert 0 <= result.rel_error < 1.0

    def test_process_exact_mode_per_tensor(self):
        """Test exact mode with group_size=0 (per-tensor, one scale for entire matrix)."""
        # Use scale=2 for entire matrix
        weight = np.array([
            [2.0, 0.0, -2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [-2.0, 2.0, -2.0, 0.0],
        ], dtype=np.float32)

        result = process_tensor("test", weight, mode="exact", group_size=0)

        assert result.is_ternary == True
        assert result.max_deviation < 1e-6
        # In exact mode with per-tensor, scales should have 1 element
        # The trits are packed based on weight/scale


class TestRoundTrip:
    """Test that quantization can be reversed (approximately)."""

    def test_quantize_then_unpack_matches(self):
        """Test that quantized weights can be reconstructed."""
        from src.tern_convert.packing import unpack_ternary_matrix

        weight = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ], dtype=np.float32)

        packed, scales, bytes_per_row = quantize_absmean(weight, group_size=0)
        trits = unpack_ternary_matrix(packed, weight.shape, bytes_per_row)

        # Reconstruct - group_size=0 means one scale per tensor
        reconstructed = trits.astype(np.float32) * scales[0]

        # Check relative error is reasonable
        diff = np.abs(reconstructed - weight)
        rel_error = np.linalg.norm(diff) / np.linalg.norm(weight)
        assert rel_error < 0.5  # Should be reasonably close