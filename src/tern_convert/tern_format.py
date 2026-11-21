"""Tern file format (.tern) reading and writing.

Format v0.1:
- Header: magic bytes b"TERN", u32 LE major, u32 LE minor, u64 LE header_len
- JSON header (UTF-8) padded with spaces so data section starts on 64-byte boundary
- Data section: tensor bytes, each tensor 64-byte aligned
- Tensor offsets in JSON are relative to start of data section
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .packing import TRITS_PER_BYTE

# Format constants
MAGIC = b"TERN"
FORMAT_VERSION_MAJOR = 0
FORMAT_VERSION_MINOR = 1
HEADER_ALIGNMENT = 64
TENSOR_ALIGNMENT = 64

# Struct formats (little-endian)
HEADER_STRUCT = struct.Struct("<4sIIQ")  # magic, major, minor, header_len
# 4 + 4 + 4 + 8 = 20 bytes


class TernTensorEntry:
    """Represents a tensor entry in the .tern file."""

    def __init__(
        self,
        name: str,
        dtype: str,
        shape: list[int],
        offset: int,
        nbytes: int,
        # For ternary tensors
        group_size: int | None = None,
        packed: dict | None = None,
        scales: dict | None = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.nbytes = nbytes
        self.group_size = group_size
        self.packed = packed
        self.scales = scales

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        if self.dtype == "tern5":
            return {
                "dtype": "tern5",
                "shape": self.shape,
                "group_size": self.group_size,
                "packed": self.packed,
                "scales": self.scales,
            }
        else:
            return {
                "dtype": self.dtype,
                "shape": self.shape,
                "offset": self.offset,
                "nbytes": self.nbytes,
            }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> TernTensorEntry:
        """Create from dict (from JSON)."""
        dtype = data["dtype"]
        if dtype == "tern5":
            return cls(
                name=name,
                dtype=dtype,
                shape=data["shape"],
                offset=0,  # Not used for tern5
                nbytes=0,  # Not used for tern5
                group_size=data["group_size"],
                packed=data["packed"],
                scales=data["scales"],
            )
        else:
            return cls(
                name=name,
                dtype=dtype,
                shape=data["shape"],
                offset=data["offset"],
                nbytes=data["nbytes"],
            )


class TernHeader:
    """Represents the .tern file header."""

    def __init__(
        self,
        format_version: str = "0.1",
        metadata: dict[str, Any] | None = None,
        tensors: dict[str, TernTensorEntry] | None = None,
    ):
        self.format_version = format_version
        self.metadata = metadata or {}
        self.tensors = tensors or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "metadata": self.metadata,
            "tensors": {name: t.to_dict() for name, t in self.tensors.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TernHeader:
        tensors = {
            name: TernTensorEntry.from_dict(name, tdata)
            for name, tdata in data.get("tensors", {}).items()
        }
        return cls(
            format_version=data.get("format_version", "0.1"),
            metadata=data.get("metadata", {}),
            tensors=tensors,
        )

    def to_json_bytes(self) -> bytes:
        """Serialize header to JSON bytes."""
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")


def align_offset(offset: int, alignment: int = TENSOR_ALIGNMENT) -> int:
    """Align offset to the given boundary."""
    return ((offset + alignment - 1) // alignment) * alignment


def write_tern_file(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    tensor_dtypes: dict[str, str],
    ternary_tensors: dict[str, dict] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TernHeader:
    """Write a .tern file.

    Args:
        path: Output file path
        tensors: Dict of tensor name -> numpy array (for non-ternary tensors)
        tensor_dtypes: Dict of tensor name -> dtype string ("f32", "f16", "bf16")
        ternary_tensors: Dict of tensor name -> {
            "packed": bytes, "scales": np.ndarray, "group_size": int, "shape": (out, in)
        }
        metadata: Additional metadata for header

    Returns:
        TernHeader that was written
    """
    path = Path(path)
    ternary_tensors = ternary_tensors or {}

    # Build header
    header_tensors = {}
    data_parts = []
    data_offset = 0

    # First pass: compute layout for non-ternary tensors
    for name, array in tensors.items():
        dtype = tensor_dtypes[name]
        nbytes = array.nbytes
        aligned_offset = align_offset(data_offset)
        # Pad if needed
        if aligned_offset > data_offset:
            pad_bytes = aligned_offset - data_offset
            data_parts.append(b"\x00" * pad_bytes)
            data_offset = aligned_offset

        header_tensors[name] = TernTensorEntry(
            name=name,
            dtype=dtype,
            shape=list(array.shape),
            offset=data_offset,
            nbytes=nbytes,
        )
        data_parts.append(array.tobytes())
        data_offset += nbytes

    # Second pass: ternary tensors
    for name, tdata in ternary_tensors.items():
        packed_bytes = tdata["packed"]
        scales = tdata["scales"]
        group_size = tdata["group_size"]
        shape = tdata["shape"]

        # Packed data
        packed_nbytes = len(packed_bytes)
        packed_offset = align_offset(data_offset)
        if packed_offset > data_offset:
            pad_bytes = packed_offset - data_offset
            data_parts.append(b"\x00" * pad_bytes)
            data_offset = packed_offset

        packed_entry = {"offset": data_offset, "nbytes": packed_nbytes}
        data_parts.append(packed_bytes)
        data_offset += packed_nbytes

        # Scales data
        scales_bytes = scales.astype(np.float32).tobytes()
        scales_nbytes = len(scales_bytes)
        scales_offset = align_offset(data_offset)
        if scales_offset > data_offset:
            pad_bytes = scales_offset - data_offset
            data_parts.append(b"\x00" * pad_bytes)
            data_offset = scales_offset

        scales_entry = {
            "offset": data_offset,
            "nbytes": scales_nbytes,
            "dtype": "f32",
            "count": len(scales),
        }
        data_parts.append(scales_bytes)
        data_offset += scales_nbytes

        header_tensors[name] = TernTensorEntry(
            name=name,
            dtype="tern5",
            shape=list(shape),
            offset=0,  # Not used
            nbytes=0,  # Not used
            group_size=group_size,
            packed=packed_entry,
            scales=scales_entry,
        )

    # Build header JSON
    header = TernHeader(
        format_version="0.1",
        metadata=metadata or {},
        tensors=header_tensors,
    )
    header_json = header.to_json_bytes()

    # Pad header to 64-byte boundary
    header_len = len(header_json)
    padded_header_len = align_offset(header_len, HEADER_ALIGNMENT)
    header_padding = b" " * (padded_header_len - header_len)
    header_json_padded = header_json + header_padding

    # Write file
    with open(path, "wb") as f:
        # Write binary header
        f.write(HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION_MAJOR,
            FORMAT_VERSION_MINOR,
            padded_header_len,
        ))
        # Write JSON header
        f.write(header_json_padded)
        # Write data section
        for part in data_parts:
            f.write(part)

    return header


def read_tern_file(path: str | Path) -> tuple[TernHeader, bytes]:
    """Read a .tern file.

    Returns:
        (header, data_section_bytes)
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Read binary header
        header_bytes = f.read(HEADER_STRUCT.size)
        if len(header_bytes) < HEADER_STRUCT.size:
            raise ValueError("File too small for header")

        magic, major, minor, header_len = HEADER_STRUCT.unpack(header_bytes)

        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")

        if major != FORMAT_VERSION_MAJOR:
            raise ValueError(f"Unsupported major version: {major}, expected {FORMAT_VERSION_MAJOR}")

        # Read JSON header
        header_json = f.read(header_len)
        if len(header_json) < header_len:
            raise ValueError("File too small for JSON header")

        # Strip trailing spaces (padding)
        header_json = header_json.rstrip(b" ")
        header_dict = json.loads(header_json.decode("utf-8"))

        header = TernHeader.from_dict(header_dict)

        # Read rest as data section
        data_section = f.read()

    return header, data_section


def get_tensor_data(header: TernHeader, data_section: bytes, name: str) -> np.ndarray:
    """Extract a tensor's data from the data section."""
    if name not in header.tensors:
        raise KeyError(f"Tensor {name!r} not found in header")

    entry = header.tensors[name]

    if entry.dtype == "tern5":
        raise ValueError(f"Use get_ternary_tensor for ternary tensor {name!r}")

    offset = entry.offset
    nbytes = entry.nbytes
    if offset + nbytes > len(data_section):
        raise ValueError(f"Tensor {name!r} data out of bounds")

    data_bytes = data_section[offset:offset + nbytes]

    # Map dtype string to numpy dtype
    dtype_map = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": np.uint16,  # Stored as uint16, need to convert
    }
    np_dtype = dtype_map.get(entry.dtype)
    if np_dtype is None:
        raise ValueError(f"Unknown dtype: {entry.dtype}")

    arr = np.frombuffer(data_bytes, dtype=np_dtype)
    arr = arr.reshape(entry.shape)

    # Convert BF16 (stored as uint16) to float32
    if entry.dtype == "bf16":
        # BF16 is upper 16 bits of float32
        arr = (arr.astype(np.uint32) << 16).view(np.float32)

    return arr


def get_ternary_tensor(
    header: TernHeader,
    data_section: bytes,
    name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract a ternary tensor's packed data and scales.

    Returns:
        (packed_bytes, scales, group_size)
    """
    if name not in header.tensors:
        raise KeyError(f"Tensor {name!r} not found in header")

    entry = header.tensors[name]

    if entry.dtype != "tern5":
        raise ValueError(f"Tensor {name!r} is not ternary (dtype={entry.dtype})")

    # Packed data
    packed_offset = entry.packed["offset"]
    packed_nbytes = entry.packed["nbytes"]
    if packed_offset + packed_nbytes > len(data_section):
        raise ValueError(f"Packed data for {name!r} out of bounds")
    packed_bytes = data_section[packed_offset:packed_offset + packed_nbytes]

    # Scales
    scales_offset = entry.scales["offset"]
    scales_nbytes = entry.scales["nbytes"]
    if scales_offset + scales_nbytes > len(data_section):
        raise ValueError(f"Scales for {name!r} out of bounds")
    scales_bytes = data_section[scales_offset:scales_offset + scales_nbytes]
    scales = np.frombuffer(scales_bytes, dtype=np.float32)

    group_size = entry.group_size

    return packed_bytes, scales, group_size


def verify_tern_file(path: str | Path, source_data: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Verify a .tern file for correctness.

    Args:
        path: Path to .tern file
        source_data: Optional source tensors to compare against (for exact mode)

    Returns:
        Dict with verification results
    """
    from .packing import validate_packed_bytes, unpack_ternary_matrix, TRITS_PER_BYTE

    header, data_section = read_tern_file(path)

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "tensors_checked": 0,
    }

    # Check each tensor
    for name, entry in header.tensors.items():
        results["tensors_checked"] += 1

        if entry.dtype == "tern5":
            # Verify ternary tensor
            packed_bytes, scales, group_size = get_ternary_tensor(header, data_section, name)

            # Check packed bytes <= 242
            packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)
            if not validate_packed_bytes(packed_arr):
                results["valid"] = False
                results["errors"].append(f"{name}: packed byte > 242")

            # Check scale count matches expectation
            out_features, in_features = entry.shape
            if group_size == 0:
                expected_scales = 1  # One scale per tensor
            else:
                expected_scales = out_features * ((in_features + group_size - 1) // group_size)

            if len(scales) != expected_scales:
                results["valid"] = False
                results["errors"].append(
                    f"{name}: scale count mismatch: got {len(scales)}, expected {expected_scales}"
                )

            # Unpack and verify trits are in {-1, 0, 1}
            bytes_per_row = (in_features + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
            try:
                trits = unpack_ternary_matrix(packed_arr, entry.shape, bytes_per_row)
                if not np.all((trits >= -1) & (trits <= 1)):
                    results["valid"] = False
                    results["errors"].append(f"{name}: unpacked trits not in {{-1,0,1}}")
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"{name}: unpack failed: {e}")

            # Check offsets are aligned
            if entry.packed["offset"] % TENSOR_ALIGNMENT != 0:
                results["warnings"].append(f"{name}: packed offset not 64-byte aligned")
            if entry.scales["offset"] % TENSOR_ALIGNMENT != 0:
                results["warnings"].append(f"{name}: scales offset not 64-byte aligned")

            # Compare with source if provided
            if source_data and name in source_data:
                # Reconstruct and compare
                bytes_per_row = (in_features + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
                trits = unpack_ternary_matrix(packed_arr, entry.shape, bytes_per_row)

                # Apply scales
                if group_size == 0:
                    # One scale per tensor
                    scale = scales[0]  # Single scale for entire tensor
                    reconstructed = trits.astype(np.float32) * scale
                else:
                    # Scales per group along input dimension
                    n_groups = (in_features + group_size - 1) // group_size
                    scales_reshaped = scales.reshape(out_features, n_groups)
                    reconstructed = np.zeros_like(trits, dtype=np.float32)
                    for i in range(out_features):
                        for g in range(n_groups):
                            start = g * group_size
                            end = min(start + group_size, in_features)
                            reconstructed[i, start:end] = trits[i, start:end] * scales_reshaped[i, g]

                source = source_data[name].astype(np.float32)
                diff = np.abs(reconstructed - source)
                max_diff = np.max(diff)
                if max_diff > 1e-5:
                    results["valid"] = False
                    results["errors"].append(
                        f"{name}: exact mode mismatch, max diff = {max_diff}"
                    )

        else:
            # Verify regular tensor
            if entry.offset % TENSOR_ALIGNMENT != 0:
                results["warnings"].append(f"{name}: offset not 64-byte aligned")

            # Check data in bounds
            if entry.offset + entry.nbytes > len(data_section):
                results["valid"] = False
                results["errors"].append(f"{name}: data out of bounds")

    return results