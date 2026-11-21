"""Command-line interface for tern-convert.

Usage:
    tern-convert convert INPUT -o OUTPUT.tern [options]
    tern-convert inspect FILE
    tern-convert verify FILE [--source SOURCE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .convert import (
    convert,
    convert_safetensors,
    convert_gguf,
    detect_format,
    ConversionReport,
)
from .tern_format import read_tern_file, verify_tern_file, TernHeader


def cmd_convert(args: argparse.Namespace) -> int:
    """Handle the 'convert' command."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    mode = args.mode
    group_size = args.group_size
    ternary_regex = args.ternary_regex
    keep_regex = args.keep_regex
    report_path = Path(args.report) if args.report else None
    max_rel_error = args.max_rel_error

    try:
        report = convert(
            input_path=input_path,
            output_path=output_path,
            mode=mode,
            group_size=group_size,
            ternary_regex=ternary_regex,
            keep_regex=keep_regex,
            report_path=report_path,
            max_rel_error=max_rel_error,
        )

        # Print summary
        print(f"Conversion complete: {output_path}")
        print(f"  Mode: {mode}")
        print(f"  Tensors: {report.summary['total_tensors']}")
        print(f"    Ternary: {report.summary['ternary_tensors']}")
        print(f"    Regular: {report.summary['regular_tensors']}")
        if report.summary['avg_rel_error'] > 0:
            print(f"  Avg rel error: {report.summary['avg_rel_error']:.6f}")

        if report_path:
            print(f"  Report: {report_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_inspect(args: argparse.Namespace) -> int:
    """Handle the 'inspect' command."""
    path = Path(args.file)

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    suffix = path.suffix.lower()

    if suffix == ".tern":
        try:
            header, data_section = read_tern_file(path)
            print(f"TERN file: {path}")
            print(f"  Version: {header.format_version}")
            print(f"  Metadata:")
            for key, value in header.metadata.items():
                if key in ("config", "tokenizer"):
                    print(f"    {key}: <present>")
                else:
                    print(f"    {key}: {value}")

            print(f"  Tensors ({len(header.tensors)}):")
            for name, entry in sorted(header.tensors.items()):
                shape = entry.shape
                if entry.dtype == "tern5":
                    print(f"    {name}: dtype={entry.dtype}, shape={shape}, group_size={entry.group_size}")
                    print(f"      packed: offset={entry.packed['offset']}, nbytes={entry.packed['nbytes']}")
                    print(f"      scales: offset={entry.scales['offset']}, nbytes={entry.scales['nbytes']}, count={entry.scales['count']}")
                else:
                    print(f"    {name}: dtype={entry.dtype}, shape={shape}, offset={entry.offset}, nbytes={entry.nbytes}")

            return 0

        except Exception as e:
            print(f"Error reading TERN file: {e}", file=sys.stderr)
            return 1

    elif suffix == ".safetensors":
        from .readers.safetensors import read_safetensors_file, read_safetensors_header

        try:
            header = read_safetensors_header(path)
            print(f"Safetensors file: {path}")
            metadata = header.get("__metadata__", {})
            if metadata:
                print(f"  Metadata:")
                for key, value in metadata.items():
                    print(f"    {key}: {value}")

            tensors = [k for k in header.keys() if k != "__metadata__"]
            print(f"  Tensors ({len(tensors)}):")
            for name in tensors[:10]:  # Show first 10
                info = header[name]
                print(f"    {name}: dtype={info['dtype']}, shape={info['shape']}")
            if len(tensors) > 10:
                print(f"    ... and {len(tensors) - 10} more")

            return 0

        except Exception as e:
            print(f"Error reading Safetensors file: {e}", file=sys.stderr)
            return 1

    elif suffix == ".gguf":
        from .readers.gguf import GGUFReader

        try:
            with GGUFReader(path) as reader:
                print(f"GGUF file: {path}")
                print(f"  Version: {reader.metadata.get('general.version', 'unknown')}")
                print(f"  Architecture: {reader.metadata.get('general.architecture', 'unknown')}")
                print(f"  Metadata:")
                for key, value in reader.metadata.items():
                    if key.startswith("general."):
                        continue
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"    {key}: {value}")

                tensor_names = reader.tensor_names
                print(f"  Tensors ({len(tensor_names)}):")
                for name in tensor_names[:10]:  # Show first 10
                    info = reader.get_tensor_info(name)
                    dims = info["dims"]
                    tensor_type = info["tensor_type"]
                    type_name = "unknown"
                    # Get type name from reader
                    from .readers.gguf import GGUF_TENSOR_TYPES
                    if tensor_type in GGUF_TENSOR_TYPES:
                        type_name = GGUF_TENSOR_TYPES[tensor_type][0]
                    print(f"    {name}: type={type_name}, dims={dims}")
                if len(tensor_names) > 10:
                    print(f"    ... and {len(tensor_names) - 10} more")

            return 0

        except Exception as e:
            print(f"Error reading GGUF file: {e}", file=sys.stderr)
            return 1

    else:
        print(f"Unknown file type: {suffix}", file=sys.stderr)
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Handle the 'verify' command."""
    path = Path(args.file)

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    suffix = path.suffix.lower()

    if suffix != ".tern":
        print(f"Verify only supports .tern files, got: {suffix}", file=sys.stderr)
        return 1

    try:
        results = verify_tern_file(path)

        print(f"Verification: {path}")
        print(f"  Valid: {results['valid']}")
        print(f"  Tensors checked: {results['tensors_checked']}")

        if results['errors']:
            print(f"  Errors:")
            for err in results['errors']:
                print(f"    - {err}")

        if results['warnings']:
            print(f"  Warnings:")
            for warn in results['warnings']:
                print(f"    - {warn}")

        # If source provided, verify exact match
        if args.source:
            source_path = Path(args.source)
            if source_path.suffix.lower() == ".safetensors":
                from .readers.safetensors import SafetensorsShardReader
                with SafetensorsShardReader(source_path) as reader:
                    source_data = {}
                    for name in reader.tensor_names():
                        source_data[name] = reader.read_tensor(name)

                # Re-verify with source comparison
                results_with_source = verify_tern_file(path, source_data)
                if not results_with_source['valid']:
                    print(f"  Source comparison FAILED:")
                    for err in results_with_source['errors']:
                        print(f"    - {err}")
                    return 1
                else:
                    print(f"  Source comparison: PASSED")

            elif source_path.suffix.lower() == ".gguf":
                from .readers.gguf import GGUFReader
                with GGUFReader(source_path) as reader:
                    source_data = {}
                    for name in reader.tensor_names():
                        source_data[name] = reader.read_tensor(name)

                results_with_source = verify_tern_file(path, source_data)
                if not results_with_source['valid']:
                    print(f"  Source comparison FAILED:")
                    for err in results_with_source['errors']:
                        print(f"    - {err}")
                    return 1
                else:
                    print(f"  Source comparison: PASSED")

        return 0 if results['valid'] else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="tern-convert",
        description="Convert GGUF and safetensors models into the .tern ternary format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a model to .tern format",
        description="Convert GGUF or safetensors format to .tern format",
    )
    convert_parser.add_argument(
        "input", help="Input file (.safetensors or .gguf) or directory of sharded safetensors"
    )
    convert_parser.add_argument(
        "-o", "--output", required=True, help="Output .tern file path"
    )
    convert_parser.add_argument(
        "--mode", choices=["exact", "quantize"], required=True,
        help="Conversion mode: 'exact' for already-ternary, 'quantize' for normal models"
    )
    convert_parser.add_argument(
        "--group-size", type=int, default=0,
        help="Scale group size (0 = one scale per tensor, default: 0)"
    )
    convert_parser.add_argument(
        "--ternary-regex", default=r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)(\.weight)?$",
        help="Regex for tensors to convert to ternary"
    )
    convert_parser.add_argument(
        "--keep-regex", default=r"$^",
        help="Regex for tensors to keep in original dtype (excludes from ternary)"
    )
    convert_parser.add_argument(
        "--report", type=str,
        help="Path to write JSON report"
    )
    convert_parser.add_argument(
        "--max-rel-error", type=float,
        help="Fail if ternary tensor's relative error exceeds this value"
    )
    convert_parser.set_defaults(func=cmd_convert)

    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a model file",
        description="Display information about .tern, .safetensors, or .gguf files"
    )
    inspect_parser.add_argument("file", help="File to inspect")
    inspect_parser.set_defaults(func=cmd_inspect)

    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a .tern file",
        description="Check .tern file integrity and optionally compare with source"
    )
    verify_parser.add_argument("file", help=".tern file to verify")
    verify_parser.add_argument(
        "--source", type=str,
        help="Source model to compare against (for exact mode)"
    )
    verify_parser.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())