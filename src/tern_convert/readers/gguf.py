"""GGUF format reader.

Format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Header: magic "GGUF", version, tensor_count, metadata_kv_count
- Metadata key-value pairs (various types including arrays)
- Tensor info table: name, n_dims, dims, dtype, offset
- Data section: tensor data with alignment (general.alignment, default 32)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGUF tensor types (subset we support natively)
GGUF_TENSOR_TYPES = {
    0: ("F32", np.float32, 4),      # F32
    1: ("F16", np.float16, 2),      # F16
    2: ("Q4_0", None, None),        # Quantized - needs gguf package
    3: ("Q4_1", None, None),
    6: ("Q8_0", None, None),
    7: ("Q5_0", None, None),
    8: ("Q5_1", None, None),
    9: ("Q2_K", None, None),
    10: ("Q3_K", None, None),
    11: ("Q4_K", None, None),
    12: ("Q5_K", None, None),
    13: ("Q6_K", None, None),
    14: ("Q8_K", None, None),
    15: ("BF16", np.uint16, 2),     # BF16 stored as uint16
    16: ("Q4_0_4_4", None, None),
    17: ("Q4_0_4_8", None, None),
    18: ("Q4_0_8_8", None, None),
    19: ("TQ1_0", None, None),      # Ternary - TODO
    20: ("TQ2_0", None, None),      # Ternary - TODO
}

# Struct formats (little-endian)
HEADER_STRUCT = struct.Struct("<4sIII")  # magic, version, tensor_count, metadata_kv_count
TENSOR_INFO_HEADER = struct.Struct("<I")  # name length


class GGUFReader:
    """Read tensors from a GGUF file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._header: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._tensor_infos: dict[str, dict] = {}
        self._data_offset: int = 0
        self._alignment: int = 32
        self._mmap: bytes | None = None
        self._file_handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open and parse the GGUF file."""
        self._file_handle = open(self.path, "rb")
        self._parse_header()
        self._parse_metadata()
        self._parse_tensor_info()
        # Read all data for simplicity (could mmap for large files)
        self._file_handle.seek(self._data_offset)
        self._mmap = self._file_handle.read()

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self._mmap = None

    def _parse_header(self):
        header_bytes = self._file_handle.read(HEADER_STRUCT.size)
        if len(header_bytes) < HEADER_STRUCT.size:
            raise ValueError("File too small for GGUF header")

        magic, version, tensor_count, metadata_kv_count = HEADER_STRUCT.unpack(header_bytes)

        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {GGUF_MAGIC!r}")

        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}, expected {GGUF_VERSION}")

        self._header = {
            "magic": magic,
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_kv_count,
        }

    def _read_string(self) -> str:
        """Read a length-prefixed string."""
        length_bytes = self._file_handle.read(4)
        if len(length_bytes) < 4:
            raise ValueError("Unexpected EOF reading string length")
        length = struct.unpack("<I", length_bytes)[0]
        string_bytes = self._file_handle.read(length)
        if len(string_bytes) < length:
            raise ValueError("Unexpected EOF reading string")
        return string_bytes.decode("utf-8")

    def _read_value(self, value_type: int) -> Any:
        """Read a single GGUF value."""
        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", self._file_handle.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", self._file_handle.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", self._file_handle.read(2))[0]
        elif value_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", self._file_handle.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return struct.unpack("<I", self._file_handle.read(4))[0]
        elif value_type == GGUF_TYPE_INT32:
            return struct.unpack("<i", self._file_handle.read(4))[0]
        elif value_type == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", self._file_handle.read(4))[0]
        elif value_type == GGUF_TYPE_BOOL:
            return bool(struct.unpack("<B", self._file_handle.read(1))[0])
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string()
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack("<I", self._file_handle.read(4))[0]
            array_len = struct.unpack("<I", self._file_handle.read(4))[0]
            return [self._read_value(array_type) for _ in range(array_len)]
        elif value_type == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", self._file_handle.read(8))[0]
        elif value_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", self._file_handle.read(8))[0]
        elif value_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", self._file_handle.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF value type: {value_type}")

    def _parse_metadata(self):
        """Parse metadata key-value pairs."""
        for _ in range(self._header["metadata_kv_count"]):
            key = self._read_string()
            value_type = struct.unpack("<I", self._file_handle.read(4))[0]
            value = self._read_value(value_type)
            self._metadata[key] = value

        # Get alignment from metadata (default 32)
        self._alignment = self._metadata.get("general.alignment", 32)

    def _parse_tensor_info(self):
        """Parse tensor info table."""
        for _ in range(self._header["tensor_count"]):
            # Name
            name = self._read_string()

            # Number of dimensions
            n_dims = struct.unpack("<I", self._file_handle.read(4))[0]

            # Dimensions (ne[0] is fastest changing / innermost)
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack("<I", self._file_handle.read(4))[0])

            # Tensor type
            tensor_type = struct.unpack("<I", self._file_handle.read(4))[0]

            # Offset in data section
            offset = struct.unpack("<Q", self._file_handle.read(8))[0]

            self._tensor_infos[name] = {
                "name": name,
                "n_dims": n_dims,
                "dims": dims,  # GGUF order: ne[0] = fastest
                "tensor_type": tensor_type,
                "offset": offset,
            }

        # Data section starts after tensor info, aligned
        self._data_offset = self._file_handle.tell()
        self._data_offset = ((self._data_offset + self._alignment - 1) // self._alignment) * self._alignment

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def tensor_names(self) -> list[str]:
        return list(self._tensor_infos.keys())

    def get_tensor_info(self, name: str) -> dict[str, Any]:
        if name not in self._tensor_infos:
            raise KeyError(f"Tensor {name!r} not found")
        return self._tensor_infos[name]

    def read_tensor(self, name: str) -> np.ndarray:
        """Read a tensor as numpy array, handling GGUF->numpy dimension reversal."""
        info = self.get_tensor_info(name)
        tensor_type = info["tensor_type"]
        dims = info["dims"]  # GGUF order: ne[0] is fastest

        # Get type info
        type_info = GGUF_TENSOR_TYPES.get(tensor_type)
        if type_info is None:
            raise ValueError(f"Unknown tensor type: {tensor_type}")

        type_name, np_dtype, type_size = type_info

        if np_dtype is None:
            # Quantized type - would need gguf package
            raise NotImplementedError(
                f"Tensor type {type_name} (id={tensor_type}) requires optional 'gguf' package. "
                f"Install with: pip install 'tern-convert[gguf]'"
            )

        # GGUF stores dimensions with ne[0] as fastest (innermost)
        # NumPy uses C-order: last dimension is fastest
        # So we need to reverse the shape for numpy
        numpy_shape = tuple(reversed(dims))

        # Calculate total elements
        n_elements = np.prod(dims)
        nbytes = n_elements * type_size

        # Read data
        offset = info["offset"]
        if offset + nbytes > len(self._mmap):
            raise ValueError(f"Tensor {name} data out of bounds")

        data_bytes = self._mmap[offset:offset + nbytes]
        if len(data_bytes) != nbytes:
            raise ValueError(f"Tensor {name}: expected {nbytes} bytes, got {len(data_bytes)}")

        # Create array
        arr = np.frombuffer(data_bytes, dtype=np_dtype)
        arr = arr.reshape(numpy_shape)

        # Convert BF16 (stored as uint16) to float32
        if tensor_type == 15:  # BF16
            arr = (arr.astype(np.uint32) << 16).view(np.float32)

        return arr


def read_gguf_file(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Convenience function to read all tensors from a GGUF file.

    Returns:
        (metadata, tensors_dict)
    """
    with GGUFReader(path) as reader:
        tensors = {name: reader.read_tensor(name) for name in reader.tensor_names}
        return reader.metadata, tensors


def write_gguf_file(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    alignment: int = 32,
):
    """Write tensors to a GGUF file (for testing).

    Only supports F32, F16, BF16 tensors.
    """
    path = Path(path)
    metadata = metadata or {}

    # Prepare tensor infos
    tensor_infos = []
    data_parts = []
    offset = 0

    for name, arr in tensors.items():
        if arr.dtype == np.float32:
            tensor_type = 0  # F32
            type_size = 4
        elif arr.dtype == np.float16:
            tensor_type = 1  # F16
            type_size = 2
        elif arr.dtype == np.uint16:
            # Assume BF16
            tensor_type = 15  # BF16
            type_size = 2
        else:
            raise ValueError(f"Unsupported dtype for GGUF writing: {arr.dtype}")

        # Ensure C-contiguous
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        # GGUF dims: reverse of numpy shape (ne[0] = fastest)
        dims = list(reversed(arr.shape))
        n_elements = np.prod(dims)
        nbytes = n_elements * type_size

        # Align offset
        aligned_offset = ((offset + alignment - 1) // alignment) * alignment
        if aligned_offset > offset:
            pad = aligned_offset - offset
            data_parts.append(b"\x00" * pad)
            offset = aligned_offset

        tensor_infos.append({
            "name": name,
            "n_dims": len(dims),
            "dims": dims,
            "tensor_type": tensor_type,
            "offset": offset,
        })

        data_parts.append(arr.tobytes())
        offset += nbytes

    # Build metadata
    # Ensure general.alignment is set
    metadata.setdefault("general.alignment", alignment)
    metadata.setdefault("general.file_type", 1)  # F16

    # Write file
    with open(path, "wb") as f:
        # Header
        f.write(HEADER_STRUCT.pack(
            GGUF_MAGIC,
            GGUF_VERSION,
            len(tensor_infos),
            len(metadata),
        ))

        # Metadata KV pairs
        for key, value in metadata.items():
            # Write key
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))
            f.write(key_bytes)

            # Write value type and value
            if isinstance(value, bool):
                f.write(struct.pack("<I", GGUF_TYPE_BOOL))
                f.write(struct.pack("<B", int(value)))
            elif isinstance(value, int):
                if value >= 0:
                    f.write(struct.pack("<I", GGUF_TYPE_UINT64))
                    f.write(struct.pack("<Q", value))
                else:
                    f.write(struct.pack("<I", GGUF_TYPE_INT64))
                    f.write(struct.pack("<q", value))
            elif isinstance(value, float):
                f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
                f.write(struct.pack("<f", value))
            elif isinstance(value, str):
                f.write(struct.pack("<I", GGUF_TYPE_STRING))
                val_bytes = value.encode("utf-8")
                f.write(struct.pack("<I", len(val_bytes)))
                f.write(val_bytes)
            elif isinstance(value, list):
                f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
                if value:
                    # Determine array element type
                    elem = value[0]
                    if isinstance(elem, int):
                        elem_type = GGUF_TYPE_INT64
                    elif isinstance(elem, float):
                        elem_type = GGUF_TYPE_FLOAT32
                    elif isinstance(elem, str):
                        elem_type = GGUF_TYPE_STRING
                    else:
                        raise ValueError(f"Unsupported array element type: {type(elem)}")
                else:
                    elem_type = GGUF_TYPE_UINT8
                f.write(struct.pack("<I", elem_type))
                f.write(struct.pack("<I", len(value)))
                for v in value:
                    if isinstance(v, int):
                        f.write(struct.pack("<q", v))
                    elif isinstance(v, float):
                        f.write(struct.pack("<f", v))
                    elif isinstance(v, str):
                        v_bytes = v.encode("utf-8")
                        f.write(struct.pack("<I", len(v_bytes)))
                        f.write(v_bytes)
            else:
                raise ValueError(f"Unsupported metadata value type: {type(value)}")

        # Tensor info table
        for info in tensor_infos:
            name_bytes = info["name"].encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", info["n_dims"]))
            for dim in info["dims"]:
                f.write(struct.pack("<I", dim))
            f.write(struct.pack("<I", info["tensor_type"]))
            f.write(struct.pack("<Q", info["offset"]))

        # Align to data section
        data_start = f.tell()
        data_start = ((data_start + alignment - 1) // alignment) * alignment
        if data_start > f.tell():
            f.write(b"\x00" * (data_start - f.tell()))

        # Data
        for part in data_parts:
            f.write(part)