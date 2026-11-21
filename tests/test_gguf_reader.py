"""Tests for GGUF reader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tern_convert.readers.gguf import (
    GGUFReader,
    read_gguf_file,
    write_gguf_file,
    GGUF_TENSOR_TYPES,
)


class TestWriteRead:
    def test_write_read_f32(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {
                "weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "bias": np.array([0.1, 0.2], dtype=np.float32),
            }

            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                names = reader.tensor_names
                assert set(names) == {"weight", "bias"}

                weight = reader.read_tensor("weight")
                np.testing.assert_array_equal(weight, tensors["weight"])

                bias = reader.read_tensor("bias")
                np.testing.assert_array_equal(bias, tensors["bias"])

    def test_write_read_f16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {
                "weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
            }

            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                weight = reader.read_tensor("weight")
                np.testing.assert_array_equal(weight, tensors["weight"])

    def test_write_read_bf16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            # Create BF16 data (store as uint16)
            f32_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            bf16_data = (f32_data.view(np.uint32) >> 16).astype(np.uint16)

            tensors = {"weight": bf16_data}
            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                info = reader.get_tensor_info("weight")
                assert info["tensor_type"] == 15  # BF16

                weight = reader.read_tensor("weight")
                np.testing.assert_allclose(weight, f32_data, rtol=1e-2)

    def test_read_all_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {
                "a": np.ones((3, 3), dtype=np.float32),
                "b": np.zeros((2, 2), dtype=np.float32),
            }

            write_gguf_file(path, tensors)
            metadata, result = read_gguf_file(path)

            assert set(result.keys()) == {"a", "b"}
            np.testing.assert_array_equal(result["a"], tensors["a"])
            np.testing.assert_array_equal(result["b"], tensors["b"])

    def test_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {"w": np.ones((2, 2), dtype=np.float32)}
            metadata = {
                "general.name": "test_model",
                "general.architecture": "llama",
                "llama.block_count": 32,
            }

            write_gguf_file(path, tensors, metadata=metadata)

            with GGUFReader(path) as reader:
                assert reader.metadata["general.name"] == "test_model"
                assert reader.metadata["general.architecture"] == "llama"
                assert reader.metadata["llama.block_count"] == 32


class TestDimensionReversal:
    """Test that GGUF dimension order (ne[0]=fastest) is correctly reversed to numpy."""

    def test_2d_matrix_reversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            # Numpy shape (out_features=4, in_features=8)
            # In GGUF: ne[0]=8 (fastest), ne[1]=4
            numpy_array = np.arange(32, dtype=np.float32).reshape(4, 8)
            tensors = {"weight": numpy_array}

            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                info = reader.get_tensor_info("weight")
                # GGUF dims should be [8, 4] (ne[0]=8, ne[1]=4)
                assert info["dims"] == [8, 4]

                # Reading should give back (4, 8)
                weight = reader.read_tensor("weight")
                assert weight.shape == (4, 8)
                np.testing.assert_array_equal(weight, numpy_array)

    def test_3d_tensor_reversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            # Numpy shape (2, 3, 4)
            # In GGUF: ne[0]=4, ne[1]=3, ne[2]=2
            numpy_array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            tensors = {"tensor": numpy_array}

            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                info = reader.get_tensor_info("tensor")
                assert info["dims"] == [4, 3, 2]

                tensor = reader.read_tensor("tensor")
                assert tensor.shape == (2, 3, 4)
                np.testing.assert_array_equal(tensor, numpy_array)

    def test_1d_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            numpy_array = np.arange(10, dtype=np.float32)
            tensors = {"vec": numpy_array}

            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                info = reader.get_tensor_info("vec")
                assert info["dims"] == [10]

                vec = reader.read_tensor("vec")
                assert vec.shape == (10,)
                np.testing.assert_array_equal(vec, numpy_array)


class TestAlignment:
    def test_default_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {"w": np.ones((2, 2), dtype=np.float32)}
            write_gguf_file(path, tensors)

            with GGUFReader(path) as reader:
                assert reader._alignment == 32

    def test_custom_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {"w": np.ones((2, 2), dtype=np.float32)}
            write_gguf_file(path, tensors, alignment=64)

            with GGUFReader(path) as reader:
                assert reader._alignment == 64


class TestErrors:
    def test_missing_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"
            write_gguf_file(path, {"w": np.ones((2, 2), dtype=np.float32)})

            with GGUFReader(path) as reader:
                with pytest.raises(KeyError):
                    reader.read_tensor("nonexistent")

    def test_unsupported_quantized_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            # Manually create a GGUF with Q4_0 tensor
            # This is complex, so just test the error path exists
            pass  # Skipped - hard to create without full writer support


class TestTensorTypes:
    def test_supported_types_in_map(self):
        assert 0 in GGUF_TENSOR_TYPES  # F32
        assert 1 in GGUF_TENSOR_TYPES  # F16
        assert 15 in GGUF_TENSOR_TYPES  # BF16

    def test_quantized_types_marked_unsupported(self):
        # These should have None for np_dtype
        assert GGUF_TENSOR_TYPES[2][1] is None  # Q4_0
        assert GGUF_TENSOR_TYPES[11][1] is None  # Q4_K
        assert GGUF_TENSOR_TYPES[19][1] is None  # TQ1_0
        assert GGUF_TENSOR_TYPES[20][1] is None  # TQ2_0


class TestMetadataTypes:
    def test_various_metadata_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"

            tensors = {"w": np.ones((2, 2), dtype=np.float32)}
            metadata = {
                "bool_val": True,
                "int_val": 42,
                "neg_int": -5,
                "float_val": 3.14,
                "str_val": "hello",
                "array_int": [1, 2, 3],
                "array_float": [1.0, 2.0],
                "array_str": ["a", "b"],
            }

            write_gguf_file(path, tensors, metadata=metadata)

            with GGUFReader(path) as reader:
                assert reader.metadata["bool_val"] is True
                assert reader.metadata["int_val"] == 42
                assert reader.metadata["neg_int"] == -5
                assert abs(reader.metadata["float_val"] - 3.14) < 1e-5
                assert reader.metadata["str_val"] == "hello"
                assert reader.metadata["array_int"] == [1, 2, 3]
                assert all(abs(a - b) < 1e-5 for a, b in zip(reader.metadata["array_float"], [1.0, 2.0]))
                assert reader.metadata["array_str"] == ["a", "b"]