"""Safetensors format reader.

Format spec: https://github.com/huggingface/safetensors
- 8-byte little-endian header size (u64)
- JSON header (UTF-8) containing tensor metadata
- Data section: raw tensor bytes
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

# Dtype mapping from safetensors to numpy
SAFETENSORS_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # Stored as uint16, convert to float32 on read
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
    "F64": np.float64,
}

# Reverse mapping for writing
NUMPY_TO_SAFETENSORS = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    # BF16: we'll detect it via metadata or allow explicit override
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint32): "U32",
    np.dtype(np.uint16): "U16",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.float64): "F64",
}

HEADER_SIZE_STRUCT = struct.Struct("<Q")  # u64 little-endian


class SafetensorsReader:
    """Read tensors from a safetensors file or directory of shards."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._header: dict[str, Any] | None = None
        self._data_offset: int = 0
        self._file_handle = None
        self._mmap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the file (or first shard) for reading."""
        if self.path.is_dir():
            # Directory of shards - find all .safetensors files
            self.shard_files = sorted(self.path.glob("*.safetensors"))
            if not self.shard_files:
                raise FileNotFoundError(f"No .safetensors files in {self.path}")
            self.current_shard = 0
            self._open_shard(self.shard_files[0])
        else:
            # Single file
            self.shard_files = [self.path]
            self.current_shard = 0
            self._open_shard(self.path)

    def _open_shard(self, shard_path: Path):
        """Open a specific shard file."""
        self.close()
        self._file_handle = open(shard_path, "rb")
        # Read header size
        header_size_bytes = self._file_handle.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"File too small for header size: {shard_path}")
        header_size = HEADER_SIZE_STRUCT.unpack(header_size_bytes)[0]

        # Read JSON header
        header_json = self._file_handle.read(header_size)
        if len(header_json) < header_size:
            raise ValueError(f"File too small for header: {shard_path}")

        self._header = json.loads(header_json.decode("utf-8"))
        # Remove __metadata__ if present
        self._header.pop("__metadata__", None)

        # Data starts after header
        self._data_offset = 8 + header_size

        # Memory map the data section for efficient reading
        self._mmap = self._file_handle.read()  # Read all data for simplicity

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self._mmap = None

    @property
    def header(self) -> dict[str, Any]:
        if self._header is None:
            self.open()
        return self._header

    def tensor_names(self) -> list[str]:
        """Get list of all tensor names in the file/shard."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor_info(self, name: str) -> dict[str, Any]:
        """Get tensor metadata (dtype, shape, data_offsets)."""
        if name not in self.header:
            raise KeyError(f"Tensor {name!r} not found in {self.path}")
        return self.header[name]

    def read_tensor(self, name: str) -> np.ndarray:
        """Read a tensor as numpy array."""
        info = self.get_tensor_info(name)
        dtype_str = info["dtype"]
        shape = info["shape"]
        data_offsets = info["data_offsets"]

        np_dtype = SAFETENSORS_DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        begin, end = data_offsets
        nbytes = end - begin

        # Read from memory-mapped data
        data_bytes = self._mmap[begin:end]
        if len(data_bytes) != nbytes:
            raise ValueError(f"Tensor {name}: expected {nbytes} bytes, got {len(data_bytes)}")

        arr = np.frombuffer(data_bytes, dtype=np_dtype)
        arr = arr.reshape(shape)

        # Convert BF16 (stored as uint16) to float32
        if dtype_str == "BF16":
            # BF16 is upper 16 bits of float32
            arr = (arr.astype(np.uint32) << 16).view(np.float32)

        return arr

    def read_tensor_zero_copy(self, name: str) -> np.ndarray:
        """Read a tensor using zero-copy (memory view) where possible.

        Note: This only works for F32, F16, I32, etc. BF16 requires conversion.
        """
        info = self.get_tensor_info(name)
        dtype_str = info["dtype"]
        shape = info["shape"]
        data_offsets = info["data_offsets"]

        np_dtype = SAFETENSORS_DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        begin, end = data_offsets

        # Create view into mmap
        arr = np.frombuffer(self._mmap, dtype=np_dtype, offset=begin, count=np.prod(shape))
        arr = arr.reshape(shape)

        if dtype_str == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)

        return arr


class SafetensorsShardReader:
    """Read tensors from multiple shard files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.shard_readers: dict[str, SafetensorsReader] = {}
        self.tensor_to_shard: dict[str, str] = {}
        self._index_tensors()

    def _index_tensors(self):
        """Build index of which tensor is in which shard."""
        if self.path.is_file():
            # Single file
            reader = SafetensorsReader(self.path)
            reader.open()
            for name in reader.tensor_names():
                self.tensor_to_shard[name] = str(self.path)
            self.shard_readers[str(self.path)] = reader
        else:
            # Directory of shards
            for shard_file in sorted(self.path.glob("*.safetensors")):
                reader = SafetensorsReader(shard_file)
                reader.open()
                for name in reader.tensor_names():
                    if name in self.tensor_to_shard:
                        raise ValueError(f"Duplicate tensor name across shards: {name}")
                    self.tensor_to_shard[name] = str(shard_file)
                self.shard_readers[str(shard_file)] = reader

    def tensor_names(self) -> list[str]:
        return list(self.tensor_to_shard.keys())

    def get_tensor_info(self, name: str) -> dict[str, Any]:
        shard_path = self.tensor_to_shard[name]
        return self.shard_readers[shard_path].get_tensor_info(name)

    def read_tensor(self, name: str) -> np.ndarray:
        shard_path = self.tensor_to_shard[name]
        return self.shard_readers[shard_path].read_tensor(name)

    def close(self):
        for reader in self.shard_readers.values():
            reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def read_safetensors_file(path: str | Path) -> dict[str, np.ndarray]:
    """Convenience function to read all tensors from a safetensors file or directory."""
    with SafetensorsShardReader(path) as reader:
        return {name: reader.read_tensor(name) for name in reader.tensor_names()}


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read just the header from a safetensors file."""
    path = Path(path)
    if path.is_dir():
        # Read first shard's header
        shard_files = sorted(path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files in {path}")
        path = shard_files[0]

    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = HEADER_SIZE_STRUCT.unpack(header_size_bytes)[0]
        header_json = f.read(header_size)
        return json.loads(header_json.decode("utf-8"))


def write_safetensors_file(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
):
    """Write tensors to a safetensors file (for testing)."""
    path = Path(path)

    # Build header
    header = {}
    data_parts = []
    offset = 0

    for name, arr in tensors.items():
        dt = np.dtype(arr.dtype)
        if dt not in NUMPY_TO_SAFETENSORS:
            raise ValueError(f"Unsupported dtype for writing: {arr.dtype}")

        dtype_str = NUMPY_TO_SAFETENSORS[dt]
        shape = list(arr.shape)
        nbytes = arr.nbytes

        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }

        # Ensure contiguous
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        data_parts.append(arr.tobytes())
        offset += nbytes

    if metadata:
        header["__metadata__"] = metadata

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(HEADER_SIZE_STRUCT.pack(header_size))
        f.write(header_json)
        for part in data_parts:
            f.write(part)