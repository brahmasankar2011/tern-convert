"""Tests for ternary packing/unpacking."""

import numpy as np
import pytest

from src.tern_convert.packing import (
    pack_ternary_row,
    unpack_ternary_row,
    pack_ternary_matrix,
    unpack_ternary_matrix,
    validate_packed_bytes,
    validate_trits,
    TRITS_PER_BYTE,
    MAX_PACKED_VALUE,
)


class TestPackUnpackRow:
    """Test packing and unpacking of single rows."""

    def test_empty_row(self):
        trits = np.array([], dtype=np.int8)
        packed = pack_ternary_row(trits)
        assert len(packed) == 0
        unpacked = unpack_ternary_row(packed, 0)
        assert len(unpacked) == 0

    def test_single_trit(self):
        for val in [-1, 0, 1]:
            trits = np.array([val], dtype=np.int8)
            packed = pack_ternary_row(trits)
            assert len(packed) == 1
            assert packed[0] <= MAX_PACKED_VALUE
            unpacked = unpack_ternary_row(packed, 1)
            assert unpacked[0] == val

    def test_five_trits_exact(self):
        trits = np.array([-1, 0, 1, -1, 1], dtype=np.int8)
        packed = pack_ternary_row(trits)
        assert len(packed) == 1
        unpacked = unpack_ternary_row(packed, 5)
        np.testing.assert_array_equal(unpacked, trits)

    def test_six_trits_padding(self):
        trits = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int8)
        packed = pack_ternary_row(trits)
        assert len(packed) == 2  # padded to 10 trits = 2 bytes
        unpacked = unpack_ternary_row(packed, 6)
        np.testing.assert_array_equal(unpacked, trits)

    def test_all_combinations_5_trits(self):
        """Test all 3^5 = 243 combinations of 5 trits."""
        from itertools import product
        for combo in product([-1, 0, 1], repeat=5):
            trits = np.array(combo, dtype=np.int8)
            packed = pack_ternary_row(trits)
            assert len(packed) == 1
            assert packed[0] <= MAX_PACKED_VALUE
            unpacked = unpack_ternary_row(packed, 5)
            np.testing.assert_array_equal(unpacked, trits)

    def test_round_trip_various_lengths(self):
        for length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 64, 100, 127, 128, 255, 256]:
            trits = np.random.randint(-1, 2, size=length, dtype=np.int8)
            packed = pack_ternary_row(trits)
            expected_bytes = (length + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
            assert len(packed) == expected_bytes
            unpacked = unpack_ternary_row(packed, length)
            np.testing.assert_array_equal(unpacked, trits)

    def test_packed_values_in_range(self):
        for length in [1, 5, 6, 10, 100]:
            trits = np.random.randint(-1, 2, size=length, dtype=np.int8)
            packed = pack_ternary_row(trits)
            assert validate_packed_bytes(packed)

    def test_known_values(self):
        # All zeros -> digit 1 -> 1*3^0 + 1*3^1 + 1*3^2 + 1*3^3 + 1*3^4 = 1+3+9+27+81 = 121
        trits = np.zeros(5, dtype=np.int8)
        packed = pack_ternary_row(trits)
        assert packed[0] == 121

        # All -1 -> digit 0 -> 0
        trits = np.full(5, -1, dtype=np.int8)
        packed = pack_ternary_row(trits)
        assert packed[0] == 0

        # All 1 -> digit 2 -> 2*(1+3+9+27+81) = 2*121 = 242
        trits = np.full(5, 1, dtype=np.int8)
        packed = pack_ternary_row(trits)
        assert packed[0] == 242


class TestPackUnpackMatrix:
    """Test packing and unpacking of 2D matrices."""

    def test_small_matrix(self):
        matrix = np.array([
            [-1, 0, 1, -1, 1],
            [1, -1, 0, 1, -1],
            [0, 1, -1, 0, 1],
        ], dtype=np.int8)

        packed, bytes_per_row = pack_ternary_matrix(matrix)
        assert bytes_per_row == 1
        assert len(packed) == 3

        unpacked = unpack_ternary_matrix(packed, matrix.shape, bytes_per_row)
        np.testing.assert_array_equal(unpacked, matrix)

    def test_matrix_width_not_multiple_of_5(self):
        matrix = np.random.randint(-1, 2, size=(4, 7), dtype=np.int8)
        packed, bytes_per_row = pack_ternary_matrix(matrix)
        assert bytes_per_row == 2  # ceil(7/5) = 2
        assert len(packed) == 8

        unpacked = unpack_ternary_matrix(packed, matrix.shape, bytes_per_row)
        np.testing.assert_array_equal(unpacked, matrix)

    def test_large_matrix(self):
        matrix = np.random.randint(-1, 2, size=(128, 256), dtype=np.int8)
        packed, bytes_per_row = pack_ternary_matrix(matrix)
        assert bytes_per_row == 52  # ceil(256/5) = 52
        assert len(packed) == 128 * 52

        unpacked = unpack_ternary_matrix(packed, matrix.shape, bytes_per_row)
        np.testing.assert_array_equal(unpacked, matrix)

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            pack_ternary_matrix(np.array([1, 2, 3], dtype=np.int8))

        with pytest.raises(ValueError):
            pack_ternary_matrix(np.zeros((2, 3, 4), dtype=np.int8))


class TestValidation:
    """Test validation functions."""

    def test_validate_packed_bytes_valid(self):
        packed = np.array([0, 121, 242, 100], dtype=np.uint8)
        assert validate_packed_bytes(packed)

    def test_validate_packed_bytes_invalid(self):
        packed = np.array([0, 243], dtype=np.uint8)  # 243 > 242
        assert not validate_packed_bytes(packed)

    def test_validate_trits_valid(self):
        trits = np.array([-1, 0, 1, -1, 0, 1], dtype=np.int8)
        assert validate_trits(trits)

    def test_validate_trits_invalid(self):
        trits = np.array([-1, 0, 2], dtype=np.int8)
        assert not validate_trits(trits)

        trits = np.array([-2, 0, 1], dtype=np.int8)
        assert not validate_trits(trits)


class TestConstants:
    def test_constants(self):
        assert TRITS_PER_BYTE == 5
        assert MAX_PACKED_VALUE == 242