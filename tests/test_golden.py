"""Tests for golden file verification.

These tests load pre-computed golden files and verify that the decode
operations produce the expected values.
"""

from pathlib import Path

import numpy as np
import pytest

from src.tern_convert.tern_format import (
    read_tern_file,
    get_tensor_data,
    get_ternary_tensor,
    verify_tern_file,
)
from src.tern_convert.packing import unpack_ternary_matrix, TRITS_PER_BYTE


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_golden_simple_loads():
    """Test that simple.tern golden file loads correctly."""
    golden_path = GOLDEN_DIR / "simple.tern"

    header, data_section = read_tern_file(golden_path)

    assert header.format_version == "0.1"
    assert "weight" in header.tensors
    assert "bias" in header.tensors

    weight = get_tensor_data(header, data_section, "weight")
    np.testing.assert_array_equal(weight, np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32))

    bias = get_tensor_data(header, data_section, "bias")
    np.testing.assert_array_equal(bias, np.array([0.5, -0.5], dtype=np.float32))


def test_golden_ternary_loads():
    """Test that ternary.tern golden file loads correctly."""
    golden_path = GOLDEN_DIR / "ternary.tern"

    header, data_section = read_tern_file(golden_path)

    assert header.format_version == "0.1"
    assert header.metadata.get("test_case") == "golden_ternary"
    assert "q_proj" in header.tensors

    packed_bytes, scales, group_size = get_ternary_tensor(header, data_section, "q_proj")

    # group_size=0 means ONE scale per TENSOR
    assert group_size == 0
    np.testing.assert_array_equal(scales, np.array([1.0], dtype=np.float32))

    # Unpack and verify trits
    shape = (4, 10)
    bytes_per_row = (shape[1] + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
    trits = unpack_ternary_matrix(np.frombuffer(packed_bytes, dtype=np.uint8), shape, bytes_per_row)

    expected_trits = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
                               [0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
                               [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
                               [-1, 1, 0, -1, 1, 0, -1, 1, 0, -1]], dtype=np.int8)
    np.testing.assert_array_equal(trits, expected_trits)


def test_golden_bf16_loads():
    """Test that bf16.tern golden file loads correctly."""
    golden_path = GOLDEN_DIR / "bf16.tern"

    header, data_section = read_tern_file(golden_path)

    bf_weight = get_tensor_data(header, data_section, "bf_weight")

    expected = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)

    # BF16 may have slight precision differences, use allclose
    np.testing.assert_allclose(bf_weight, expected, rtol=1e-3)


def test_golden_simple_verify():
    """Test that verify works on golden file."""
    golden_path = GOLDEN_DIR / "simple.tern"

    result = verify_tern_file(golden_path)

    assert result["valid"]
    assert len(result["errors"]) == 0


def test_golden_ternary_verify():
    """Test that verify works on ternary golden file."""
    golden_path = GOLDEN_DIR / "ternary.tern"

    result = verify_tern_file(golden_path)

    assert result["valid"]
    assert len(result["errors"]) == 0


def test_golden_file_with_exact_mode():
    """Test golden ternary file with exact mode verification."""
    golden_path = GOLDEN_DIR / "ternary.tern"

    header, data_section = read_tern_file(golden_path)
    packed_bytes, scales, group_size = get_ternary_tensor(header, data_section, "q_proj")

    # Shape is (4, 10)
    shape = (4, 10)
    bytes_per_row = (shape[1] + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
    trits = unpack_ternary_matrix(np.frombuffer(packed_bytes, dtype=np.uint8), shape, bytes_per_row)

    # Reconstruct: trits * scale (group_size=0 means one scale per TENSOR)
    reconstructed = trits.astype(np.float32) * scales[0]

    # Source data that should match exact reconstruction with scale = 1.0
    source_data = trits.astype(np.float32) * 1.0

    result = verify_tern_file(golden_path, source_data={"q_proj": source_data})

    assert result["valid"]
    assert len(result["errors"]) == 0


def test_golden_mixed():
    """Test mixed.tern with both regular and ternary tensors."""
    golden_path = GOLDEN_DIR / "mixed.tern"

    header, data_section = read_tern_file(golden_path)

    assert header.metadata.get("test_case") == "golden_mixed"
    assert "attention_norm" in header.tensors
    assert "ffn_gate" in header.tensors

    # Check regular tensor
    norm = get_tensor_data(header, data_section, "attention_norm")
    np.testing.assert_array_equal(norm, np.ones(128, dtype=np.float32))

    # Check ternary tensor
    packed_bytes, scales, group_size = get_ternary_tensor(header, data_section, "ffn_gate")

    # group_size=0 means ONE scale per TENSOR
    assert group_size == 0
    np.testing.assert_array_equal(scales, np.array([1.0], dtype=np.float32))

    # Verify the file
    result = verify_tern_file(golden_path)
    assert result["valid"]


# Test data constants for documentation
GOLDEN_SIMPLE_WEIGHT = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
GOLDEN_SIMPLE_BIAS = np.array([0.5, -0.5], dtype=np.float32)

GOLDEN_TERNARY_TRITS = np.array([
    [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
    [0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
    [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
    [-1, 1, 0, -1, 1, 0, -1, 1, 0, -1]
], dtype=np.int8)

# Per-tensor scale (group_size=0 means one scale for entire tensor)
GOLDEN_TERNARY_SCALES = np.array([1.0], dtype=np.float32)