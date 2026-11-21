"""tern-convert: Convert GGUF and safetensors models into the .tern ternary format."""

__version__ = "0.1.0"

from .convert import (
    convert,
    ConversionReport,
    convert_safetensors,
    convert_gguf,
    detect_format,
    should_convert_to_ternary,
    DEFAULT_TERNARY_REGEX,
    DEFAULT_KEEP_REGEX,
)
from .packing import pack_ternary_matrix, unpack_ternary_matrix, TRITS_PER_BYTE
from .tern_format import (
    write_tern_file,
    read_tern_file,
    TernHeader,
    get_tensor_data,
    get_ternary_tensor,
    verify_tern_file,
)
from .readers.safetensors import write_safetensors_file, read_safetensors_header
from .readers.gguf import write_gguf_file
from .quantize import process_tensor, QuantizeResult, check_exact_ternary

__all__ = [
    # Version
    "__version__",
    # Main conversion
    "convert",
    "ConversionReport",
    "convert_safetensors",
    "convert_gguf",
    "detect_format",
    "should_convert_to_ternary",
    # Default patterns
    "DEFAULT_TERNARY_REGEX",
    "DEFAULT_KEEP_REGEX",
    # Packing/unpacking
    "pack_ternary_matrix",
    "unpack_ternary_matrix",
    "TRITS_PER_BYTE",
    # Tern file format
    "write_tern_file",
    "read_tern_file",
    "TernHeader",
    "get_tensor_data",
    "get_ternary_tensor",
    "verify_tern_file",
    # Safetensors reader/writer
    "write_safetensors_file",
    "read_safetensors_header",
    "SafetensorsReader",
    "SafetensorsShardReader",
    # GGUF writer/reader
    "write_gguf_file",
    "GGUFReader",
    "read_gguf_file",
    # Quantization
    "process_tensor",
    "QuantizeResult",
    "check_exact_ternary",
]