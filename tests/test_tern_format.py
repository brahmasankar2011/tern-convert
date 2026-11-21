"""Tests for .tern file format."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tern_convert.tern_format import (
    write_tern_file,
    read_tern_file,
    get_tensor_data,
    get_ternary_tensor,
    verify_tern_file,
    align_offset,
    TENSOR_ALIGNMENT,
    HEADER_ALIGNMENT,
    TernHeader,
)
from src.tern_convert.packing import pack_ternary_matrix


class TestAlignment:
    def test_align_offset(self):
        assert align_offset(0) == 0
        assert align_offset(1) == 64
        assert align_offset(63) == 64
        assert align_offset(64) == 64
        assert align_offset(65) == 128
        assert align_offset(100, 32) == 128


class TestTernHeader:
    def test_header_roundtrip(self):
        header = TernHeader(
            format_version="0.1",
            metadata={"source": "test", "version": "1.0"},
            tensors={},
        )
        json_bytes = header.to_json_bytes()
        parsed = json.loads(json_bytes)
        assert parsed["format_version"] == "0.1"
        assert parsed["metadata"]["source"] == "test"

        header2 = TernHeader.from_dict(parsed)
        assert header2.format_version == "0.1"
        assert header2.metadata["source"] == "test"


class TestWriteReadBasic:
    def test_write_read_simple_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            tensors = {
                "weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "bias": np.array([0.1, 0.2], dtype=np.float32),
            }
            tensor_dtypes = {
                "weight": "f32",
                "bias": "f32",
            }

            header = write_tern_file(path, tensors, tensor_dtypes, metadata={"test": "value"})

            # Read back
            header2, data_section = read_tern_file(path)

            assert header2.metadata["test"] == "value"
            assert "weight" in header2.tensors
            assert "bias" in header2.tensors

            weight = get_tensor_data(header2, data_section, "weight")
            np.testing.assert_array_equal(weight, tensors["weight"])

            bias = get_tensor_data(header2, data_section, "bias")
            np.testing.assert_array_equal(bias, tensors["bias"])

    def test_write_read_f16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            tensors = {
                "weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
            }
            tensor_dtypes = {
                "weight": "f16",
            }

            header = write_tern_file(path, tensors, tensor_dtypes)
            header2, data_section = read_tern_file(path)

            weight = get_tensor_data(header2, data_section, "weight")
            np.testing.assert_array_equal(weight, tensors["weight"])

    def test_write_read_bf16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            # Create BF16 data (simulate by converting from float32)
            f32_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            # Convert to BF16 representation (upper 16 bits)
            bf16_data = (f32_data.view(np.uint32) >> 16).astype(np.uint16)

            tensors = {"weight": bf16_data}
            tensor_dtypes = {"weight": "bf16"}

            header = write_tern_file(path, tensors, tensor_dtypes)
            header2, data_section = read_tern_file(path)

            weight = get_tensor_data(header2, data_section, "weight")
            np.testing.assert_array_equal(weight, f32_data)

    def test_header_alignment(self):
        """Test that header is padded to 64-byte boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            tensors = {"w": np.zeros((10, 10), dtype=np.float32)}
            tensor_dtypes = {"w": "f32"}

            write_tern_file(path, tensors, tensor_dtypes)

            with open(path, "rb") as f:
                # Read binary header
                header_bytes = f.read(20)  # 4+4+4+8
                magic, major, minor, header_len = np.frombuffer(header_bytes, dtype=[
                    ('magic', 'S4'), ('major', '<u4'), ('minor', '<u4'), ('header_len', '<u8')
                ])[0]

                assert header_len % HEADER_ALIGNMENT == 0
                # Read JSON header
                json_bytes = f.read(header_len)
                # Should end with spaces
                assert json_bytes.endswith(b" ")


class TestTernaryTensors:
    def test_write_read_ternary_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            # Create a ternary weight matrix
            shape = (4, 10)  # out=4, in=10
            trits = np.random.randint(-1, 2, size=shape, dtype=np.int8)
            packed, bytes_per_row = pack_ternary_matrix(trits)

            # Scales: one per tensor (group_size=0)
            scales = np.array([np.random.uniform(0.5, 2.0)], dtype=np.float32)

            ternary_tensors = {
                "attn_q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            # Regular tensors
            tensors = {"bias": np.zeros(4, dtype=np.float32)}
            tensor_dtypes = {"bias": "f32"}

            header = write_tern_file(path, tensors, tensor_dtypes, ternary_tensors=ternary_tensors)
            header2, data_section = read_tern_file(path)

            # Read ternary tensor
            packed_bytes, scales_read, group_size = get_ternary_tensor(header2, data_section, "attn_q")

            assert group_size == 0
            np.testing.assert_array_equal(np.frombuffer(packed_bytes, dtype=np.uint8), packed)
            np.testing.assert_array_equal(scales_read, scales)

    def test_write_read_ternary_tensor_with_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (4, 20)  # out=4, in=20
            group_size = 8
            trits = np.random.randint(-1, 2, size=shape, dtype=np.int8)
            packed, bytes_per_row = pack_ternary_matrix(trits)

            # Scales: one per group along input dimension
            n_groups = (shape[1] + group_size - 1) // group_size
            scales = np.random.uniform(0.5, 2.0, size=(shape[0], n_groups)).astype(np.float32)

            ternary_tensors = {
                "ffn_gate": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": group_size,
                    "shape": shape,
                }
            }

            tensors = {}
            tensor_dtypes = {}

            header = write_tern_file(path, tensors, tensor_dtypes, ternary_tensors=ternary_tensors)
            header2, data_section = read_tern_file(path)

            packed_bytes, scales_read, gs = get_ternary_tensor(header2, data_section, "ffn_gate")

            assert gs == group_size
            np.testing.assert_array_equal(np.frombuffer(packed_bytes, dtype=np.uint8), packed)
            np.testing.assert_array_equal(scales_read, scales.flatten())  # Stored flattened


class TestVerify:
    def test_verify_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            tensors = {"w": np.array([[1.0, 2.0]], dtype=np.float32)}
            tensor_dtypes = {"w": "f32"}

            write_tern_file(path, tensors, tensor_dtypes)
            result = verify_tern_file(path)

            assert result["valid"]
            assert len(result["errors"]) == 0

    def test_verify_ternary_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (4, 10)
            trits = np.random.randint(-1, 2, size=shape, dtype=np.int8)
            packed, _ = pack_ternary_matrix(trits)
            # group_size=0 means one scale per tensor
            scales = np.ones(1, dtype=np.float32)

            ternary_tensors = {
                "q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            write_tern_file(path, {}, {}, ternary_tensors=ternary_tensors)
            result = verify_tern_file(path)

            assert result["valid"]
            assert len(result["errors"]) == 0

    def test_verify_corrupted_packed_byte(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (2, 5)
            trits = np.zeros(shape, dtype=np.int8)
            packed, _ = pack_ternary_matrix(trits)
            # Corrupt: set byte to 255 (> 242)
            packed[0] = 255
            scales = np.ones(2, dtype=np.float32)

            ternary_tensors = {
                "q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            write_tern_file(path, {}, {}, ternary_tensors=ternary_tensors)
            result = verify_tern_file(path)

            assert not result["valid"]
            assert any("packed byte > 242" in e for e in result["errors"])

    def test_verify_scale_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (2, 10)
            trits = np.zeros(shape, dtype=np.int8)
            packed, _ = pack_ternary_matrix(trits)
            # Wrong scale count: 3 instead of 1 (for group_size=0, per-tensor)
            scales = np.ones(3, dtype=np.float32)

            ternary_tensors = {
                "q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            write_tern_file(path, {}, {}, ternary_tensors=ternary_tensors)
            result = verify_tern_file(path)

            assert not result["valid"]
            assert any("scale count mismatch" in e for e in result["errors"])

    def test_verify_exact_mode_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (4, 10)
            trits = np.random.randint(-1, 2, size=shape, dtype=np.int8)
            packed, _ = pack_ternary_matrix(trits)
            # group_size=0 means one scale per tensor
            scales = np.ones(1, dtype=np.float32)

            ternary_tensors = {
                "q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            # Source data that matches exactly - per-tensor scale
            source = trits.astype(np.float32) * scales[0]

            write_tern_file(path, {}, {}, ternary_tensors=ternary_tensors)
            result = verify_tern_file(path, source_data={"q": source})

            assert result["valid"]

    def test_verify_exact_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (4, 10)
            trits = np.zeros(shape, dtype=np.int8)
            packed, _ = pack_ternary_matrix(trits)
            # group_size=0 means one scale per tensor
            scales = np.ones(1, dtype=np.float32)

            ternary_tensors = {
                "q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            # Source data that doesn't match
            source = np.ones(shape, dtype=np.float32) * 5.0

            write_tern_file(path, {}, {}, ternary_tensors=ternary_tensors)
            result = verify_tern_file(path, source_data={"q": source})

            assert not result["valid"]
            assert any("exact mode mismatch" in e for e in result["errors"])


class TestTensorAlignment:
    def test_tensor_offsets_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            tensors = {
                "a": np.zeros(10, dtype=np.float32),  # 40 bytes
                "b": np.zeros(20, dtype=np.float32),  # 80 bytes
            }
            tensor_dtypes = {"a": "f32", "b": "f32"}

            write_tern_file(path, tensors, tensor_dtypes)
            header, data_section = read_tern_file(path)

            # Both should be 64-byte aligned
            assert header.tensors["a"].offset % TENSOR_ALIGNMENT == 0
            assert header.tensors["b"].offset % TENSOR_ALIGNMENT == 0

    def test_ternary_tensor_offsets_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tern"

            shape = (4, 10)
            trits = np.zeros(shape, dtype=np.int8)
            packed, _ = pack_ternary_matrix(trits)
            # group_size=0 means one scale per tensor
            scales = np.ones(1, dtype=np.float32)

            ternary_tensors = {
                "q": {
                    "packed": packed.tobytes(),
                    "scales": scales,
                    "group_size": 0,
                    "shape": shape,
                }
            }

            write_tern_file(path, {}, {}, ternary_tensors=ternary_tensors)
            header, _ = read_tern_file(path)

            # Packed and scales should be aligned
            assert header.tensors["q"].packed["offset"] % TENSOR_ALIGNMENT == 0
            assert header.tensors["q"].scales["offset"] % TENSOR_ALIGNMENT == 0