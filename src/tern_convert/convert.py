"""Conversion orchestration: tensor selection, mode handling, report generation."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .readers.safetensors import SafetensorsShardReader, read_safetensors_header
from .readers.gguf import GGUFReader
from .tern_format import write_tern_file, TernHeader
from .quantize import process_tensor, QuantizeResult, check_exact_ternary
from .packing import pack_ternary_matrix


# Default patterns for ternary conversion
DEFAULT_TERNARY_REGEX = (
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
    r"attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)"
    r"(\.weight)?$"
)

DEFAULT_KEEP_REGEX = r"$^"  # Never matches (keep nothing by default)


@dataclass
class ConversionReport:
    """Report for a conversion run."""
    tool_version: str
    source_format: str
    source_files: list[str]
    source_sha256: list[str]
    mode: str
    group_size: int
    ternary_regex: str
    keep_regex: str
    tensors: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_json(self) -> str:
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return json.dumps(convert(asdict(self)), indent=2)


def compute_sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def should_convert_to_ternary(
    name: str,
    shape: tuple[int, ...],
    ternary_regex: str,
    keep_regex: str,
) -> bool:
    """Determine if a tensor should be converted to ternary.

    Args:
        name: Tensor name
        shape: Tensor shape
        ternary_regex: Regex for tensors to convert to ternary
        keep_regex: Regex for tensors to keep as-is (takes precedence)

    Returns:
        True if tensor should be ternary
    """
    # Only 2D matrices can be ternary
    if len(shape) != 2:
        return False

    # Check keep_regex first (exclude from ternary)
    if re.search(keep_regex, name):
        return False

    # Check ternary_regex
    if re.search(ternary_regex, name):
        return True

    return False


def convert_safetensors(
    input_path: Path,
    output_path: Path,
    mode: str,
    group_size: int,
    ternary_regex: str,
    keep_regex: str,
    report_path: Path | None = None,
    max_rel_error: float | None = None,
    tool_version: str = "0.1.0",
) -> ConversionReport:
    """Convert safetensors to .tern format."""
    source_files = [str(input_path)]
    source_sha256 = [compute_sha256(input_path)]

    # Read config.json and tokenizer.json if present
    config_data = {}
    tokenizer_text = ""
    if input_path.is_dir():
        config_file = input_path / "config.json"
        tokenizer_file = input_path / "tokenizer.json"
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
        if tokenizer_file.exists():
            with open(tokenizer_file) as f:
                tokenizer_text = f.read()
    else:
        # Single file - check for config/tokenizer in same directory
        parent = input_path.parent
        config_file = parent / "config.json"
        tokenizer_file = parent / "tokenizer.json"
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
        if tokenizer_file.exists():
            with open(tokenizer_file) as f:
                tokenizer_text = f.read()

    # Read all tensors
    with SafetensorsShardReader(input_path) as reader:
        tensor_names = reader.tensor_names()

        # First pass: determine which tensors are ternary
        ternary_tensors = {}
        regular_tensors = {}
        regular_dtypes = {}
        report_tensors = []

        for name in tensor_names:
            info = reader.get_tensor_info(name)
            shape = tuple(info["shape"])
            dtype = info["dtype"]

            # Map safetensors dtype to our dtype
            dtype_map = {
                "F32": "f32",
                "F16": "f16",
                "BF16": "bf16",
                "I32": "i32",
                "I16": "i16",
                "I8": "i8",
                "U32": "u32",
                "U16": "u16",
                "U8": "u8",
            }
            our_dtype = dtype_map.get(dtype, dtype.lower())

            is_ternary = should_convert_to_ternary(name, shape, ternary_regex, keep_regex)

            # Read tensor
            weight = reader.read_tensor(name)

            if is_ternary:
                result = process_tensor(name, weight, mode, group_size)
                report_tensors.append({
                    "name": name,
                    "shape": list(shape),
                    "original_dtype": result.original_dtype,
                    "group_size": result.group_size,
                    "is_ternary": True,
                    "max_deviation": result.max_deviation,
                    "rel_error": result.rel_error,
                    "zero_fraction": result.zero_fraction,
                })

                if mode == "exact":
                    if not result.is_ternary:
                        raise ValueError(
                            f"Tensor {name} is not ternary (max deviation: {result.max_deviation}). "
                            f"Use --mode quantize to quantize it."
                        )
                    # Compute scales per row/group and pack trits as weight / scale
                    out_features, in_features = weight.shape
                    if group_size == 0:
                        # One scale per tensor
                        scale = np.max(np.abs(weight))
                        scale = scale if scale > 0 else 1.0
                        scales = np.array([scale], dtype=np.float32)
                        # Divide entire matrix by the scale
                        trits = (weight / scale).round().clip(-1, 1).astype(np.int8)
                    else:
                        n_groups = (in_features + group_size - 1) // group_size
                        scales = np.zeros((out_features, n_groups), dtype=np.float32)
                        for i in range(out_features):
                            for g in range(n_groups):
                                start = g * group_size
                                end = min(start + group_size, in_features)
                                group = weight[i, start:end]
                                scale = np.max(np.abs(group))
                                scales[i, g] = scale if scale > 0 else 1.0
                        scales = scales.flatten()
                        # Divide each group by its scale
                        trits = np.zeros_like(weight, dtype=np.int8)
                        for i in range(out_features):
                            for g in range(n_groups):
                                start = g * group_size
                                end = min(start + group_size, in_features)
                                trits[i, start:end] = (weight[i, start:end] / scales[i * n_groups + g]).round().clip(-1, 1).astype(np.int8)

                    packed, bytes_per_row = pack_ternary_matrix(trits)

                    ternary_tensors[name] = {
                        "packed": packed.tobytes(),
                        "scales": scales,
                        "group_size": group_size,
                        "shape": shape,
                    }
                else:
                    # Quantize mode
                    ternary_tensors[name] = {
                        "packed": result.packed.tobytes(),
                        "scales": result.scales,
                        "group_size": group_size,
                        "shape": shape,
                    }

                # Check max_rel_error
                if max_rel_error is not None and result.rel_error is not None:
                    if result.rel_error > max_rel_error:
                        raise ValueError(
                            f"Tensor {name} relative error {result.rel_error:.6f} "
                            f"exceeds max_rel_error {max_rel_error}"
                        )
            else:
                regular_tensors[name] = weight
                regular_dtypes[name] = our_dtype
                report_tensors.append({
                    "name": name,
                    "shape": list(shape),
                    "original_dtype": our_dtype,
                    "group_size": 0,
                    "is_ternary": False,
                    "rel_error": None,
                    "zero_fraction": None,
                })

        # Build metadata
        metadata = {
            "source_format": "safetensors",
            "source_files": source_files,
            "source_sha256": source_sha256,
            "tool_version": tool_version,
            "config": config_data,
        }
        if tokenizer_text:
            metadata["tokenizer"] = tokenizer_text

        # Write .tern file
        header = write_tern_file(
            output_path,
            regular_tensors,
            regular_dtypes,
            ternary_tensors=ternary_tensors,
            metadata=metadata,
        )

        # Summary
        n_ternary = sum(1 for t in report_tensors if t.get("is_ternary"))
        n_regular = len(report_tensors) - n_ternary
        avg_rel_error = np.mean([
            t["rel_error"] for t in report_tensors
            if t["rel_error"] is not None
        ]) if any(t["rel_error"] is not None for t in report_tensors) else 0.0

        summary = {
            "total_tensors": len(report_tensors),
            "ternary_tensors": n_ternary,
            "regular_tensors": n_regular,
            "avg_rel_error": float(avg_rel_error),
        }

        report = ConversionReport(
            tool_version=tool_version,
            source_format="safetensors",
            source_files=source_files,
            source_sha256=source_sha256,
            mode=mode,
            group_size=group_size,
            ternary_regex=ternary_regex,
            keep_regex=keep_regex,
            tensors=report_tensors,
            summary=summary,
        )

        if report_path:
            report_path.write_text(report.to_json())

        return report


def convert_gguf(
    input_path: Path,
    output_path: Path,
    mode: str,
    group_size: int,
    ternary_regex: str,
    keep_regex: str,
    report_path: Path | None = None,
    max_rel_error: float | None = None,
    tool_version: str = "0.1.0",
) -> ConversionReport:
    """Convert GGUF to .tern format."""
    source_files = [str(input_path)]
    source_sha256 = [compute_sha256(input_path)]

    with GGUFReader(input_path) as reader:
        metadata = reader.metadata
        tensor_names = reader.tensor_names

        ternary_tensors = {}
        regular_tensors = {}
        regular_dtypes = {}
        report_tensors = []

        for name in tensor_names:
            info = reader.get_tensor_info(name)
            shape = tuple(reversed(info["dims"]))  # Reverse GGUF dims to numpy order
            tensor_type = info["tensor_type"]

            # Map GGUF tensor type to our dtype
            type_map = {
                0: "f32",
                1: "f16",
                15: "bf16",
            }
            our_dtype = type_map.get(tensor_type, f"gguf_{tensor_type}")

            is_ternary = should_convert_to_ternary(name, shape, ternary_regex, keep_regex)

            # Read tensor
            weight = reader.read_tensor(name)

            if is_ternary:
                result = process_tensor(name, weight, mode, group_size)
                report_tensors.append({
                    "name": name,
                    "shape": list(shape),
                    "original_dtype": result.original_dtype,
                    "group_size": result.group_size,
                    "is_ternary": True,
                    "max_deviation": result.max_deviation,
                    "rel_error": result.rel_error,
                    "zero_fraction": result.zero_fraction,
                })

                if mode == "exact":
                    if not result.is_ternary:
                        raise ValueError(
                            f"Tensor {name} is not ternary (max deviation: {result.max_deviation}). "
                            f"Use --mode quantize to quantize it."
                        )
                    # Compute scales per row/group and pack trits as weight / scale
                    out_features, in_features = weight.shape
                    if group_size == 0:
                        # One scale per tensor
                        scale = np.max(np.abs(weight))
                        scale = scale if scale > 0 else 1.0
                        scales = np.array([scale], dtype=np.float32)
                        # Divide entire matrix by the scale
                        trits = (weight / scale).round().clip(-1, 1).astype(np.int8)
                    else:
                        n_groups = (in_features + group_size - 1) // group_size
                        scales = np.zeros((out_features, n_groups), dtype=np.float32)
                        for i in range(out_features):
                            for g in range(n_groups):
                                start = g * group_size
                                end = min(start + group_size, in_features)
                                group = weight[i, start:end]
                                scale = np.max(np.abs(group))
                                scales[i, g] = scale if scale > 0 else 1.0
                        scales = scales.flatten()
                        # Divide each group by its scale
                        trits = np.zeros_like(weight, dtype=np.int8)
                        for i in range(out_features):
                            for g in range(n_groups):
                                start = g * group_size
                                end = min(start + group_size, in_features)
                                trits[i, start:end] = (weight[i, start:end] / scales[i * n_groups + g]).round().clip(-1, 1).astype(np.int8)

                    packed, _ = pack_ternary_matrix(trits)
                    ternary_tensors[name] = {
                        "packed": packed.tobytes(),
                        "scales": scales,
                        "group_size": group_size,
                        "shape": shape,
                    }
                else:
                    ternary_tensors[name] = {
                        "packed": result.packed.tobytes(),
                        "scales": result.scales,
                        "group_size": group_size,
                        "shape": shape,
                    }

                if max_rel_error is not None and result.rel_error is not None:
                    if result.rel_error > max_rel_error:
                        raise ValueError(
                            f"Tensor {name} relative error {result.rel_error:.6f} "
                            f"exceeds max_rel_error {max_rel_error}"
                        )
            else:
                regular_tensors[name] = weight
                regular_dtypes[name] = our_dtype
                report_tensors.append({
                    "name": name,
                    "shape": list(shape),
                    "original_dtype": our_dtype,
                    "group_size": 0,
                    "is_ternary": False,
                    "rel_error": None,
                    "zero_fraction": None,
                })

        # Build metadata
        tern_metadata = {
            "source_format": "gguf",
            "source_files": source_files,
            "source_sha256": source_sha256,
            "tool_version": tool_version,
            "gguf_metadata": metadata,
        }

        header = write_tern_file(
            output_path,
            regular_tensors,
            regular_dtypes,
            ternary_tensors=ternary_tensors,
            metadata=tern_metadata,
        )

        n_ternary = sum(1 for t in report_tensors if t.get("is_ternary"))
        n_regular = len(report_tensors) - n_ternary
        avg_rel_error = np.mean([
            t["rel_error"] for t in report_tensors
            if t["rel_error"] is not None
        ]) if any(t["rel_error"] is not None for t in report_tensors) else 0.0

        summary = {
            "total_tensors": len(report_tensors),
            "ternary_tensors": n_ternary,
            "regular_tensors": n_regular,
            "avg_rel_error": float(avg_rel_error),
        }

        report = ConversionReport(
            tool_version=tool_version,
            source_format="gguf",
            source_files=source_files,
            source_sha256=source_sha256,
            mode=mode,
            group_size=group_size,
            ternary_regex=ternary_regex,
            keep_regex=keep_regex,
            tensors=report_tensors,
            summary=summary,
        )

        if report_path:
            report_path.write_text(report.to_json())

        return report


def detect_format(input_path: Path) -> str:
    """Detect input format from file extension."""
    if input_path.is_dir():
        # Check for safetensors files
        if list(input_path.glob("*.safetensors")):
            return "safetensors"
        raise ValueError(f"Directory {input_path} contains no .safetensors files")

    suffix = input_path.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    elif suffix == ".gguf":
        return "gguf"
    else:
        raise ValueError(f"Unknown format for {input_path}. Expected .safetensors or .gguf")


def convert(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "quantize",
    group_size: int = 0,
    ternary_regex: str = DEFAULT_TERNARY_REGEX,
    keep_regex: str = DEFAULT_KEEP_REGEX,
    report_path: str | Path | None = None,
    max_rel_error: float | None = None,
    tool_version: str = "0.1.0",
) -> ConversionReport:
    """Main conversion entry point.

    Args:
        input_path: Path to .safetensors file/dir or .gguf file
        output_path: Path to output .tern file
        mode: "exact" or "quantize"
        group_size: Group size for quantization (0 = per-tensor, one scale for entire tensor)
        ternary_regex: Regex for tensors to convert to ternary
        keep_regex: Regex for tensors to keep as-is
        report_path: Optional path to write JSON report
        max_rel_error: Optional max relative error for quantize mode
        tool_version: Version string for metadata

    Returns:
        ConversionReport
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if report_path:
        report_path = Path(report_path)

    fmt = detect_format(input_path)

    if fmt == "safetensors":
        return convert_safetensors(
            input_path, output_path, mode, group_size,
            ternary_regex, keep_regex, report_path, max_rel_error, tool_version
        )
    elif fmt == "gguf":
        return convert_gguf(
            input_path, output_path, mode, group_size,
            ternary_regex, keep_regex, report_path, max_rel_error, tool_version
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")