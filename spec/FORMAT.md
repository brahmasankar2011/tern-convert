# Tern File Format v0.1

This document specifies the `.tern` file format used by the Tern inference engine.

## Overview

The `.tern` format stores model weights in a single file with a JSON header and binary data section. It supports both regular floating-point tensors (F32, F16, BF16) and ternary tensors (`tern5`).

## File Structure

```
+------------------+
| Magic "TERN"     | 4 bytes
+------------------+
| Version Major    | 4 bytes (u32 LE)
+------------------+
| Version Minor    | 4 bytes (u32 LE)
+------------------+
| Header Length    | 8 bytes (u64 LE)
+------------------+
| JSON Header      | Header Length bytes (padded to 64-byte boundary)
+------------------+
| Data Section     | Remaining bytes
+------------------+
```

All multi-byte integers are little-endian.

### Header Alignment

- The data section starts at a 64-byte boundary
- The JSON header is padded with spaces (`0x20`) to achieve this alignment

### Tensor Alignment

- Each tensor in the data section starts at a 64-byte boundary
- Offsets in the JSON header are relative to the start of the data section

## JSON Header

```json
{
  "format_version": "0.1",
  "metadata": {
    "source_format": "safetensors" | "gguf",
    "source_files": ["path/to/model.safetensors"],
    "source_sha256": ["abc123..."],
    "tool_version": "0.1.0",
    "hf_config": {...},
    "tokenizer_json": "..."
  },
  "tensors": {
    "tensor_name": { ... }
  }
}
```

## Tensor Entries

### Regular Tensor (F32, F16, BF16)

```json
{
  "dtype": "f32" | "f16" | "bf16",
  "shape": [out_features, in_features],
  "offset": 0,
  "nbytes": 33554432
}
```

### Ternary Tensor (tern5)

```json
{
  "dtype": "tern5",
  "shape": [out_features, in_features],
  "group_size": 128,
  "packed": {
    "offset": 33554432,
    "nbytes": 3358720
  },
  "scales": {
    "offset": 36913152,
    "nbytes": 524288,
    "dtype": "f32",
    "count": 131072
  }
}
```

## Packing: 5 Trits per Byte

The `tern5` type encodes ternary values (-1, 0, 1) using 5 trits per byte.

### Encoding

1. Each trit is mapped to a digit: `-1 → 0`, `0 → 1`, `+1 → 2`
2. A byte value is computed: `byte = d0 + 3×d1 + 9×d2 + 27×d3 + 81×d4`
3. Valid byte values range from 0 to 242

### Example

| Trits | Digits | Byte Value |
|-------|--------|------------|
| -1, 0, 1, -1, 1 | 0, 1, 2, 0, 2 | 0 + 3 + 18 + 0 + 162 = 183 |
| 0, 0, 0, 0, 0 | 1, 1, 1, 1, 1 | 1 + 3 + 9 + 27 + 81 = 121 |
| -1, -1, -1, -1, -1 | 0, 0, 0, 0, 0 | 0 |
| 1, 1, 1, 1, 1 | 2, 2, 2, 2, 2 | 2×121 = 242 |

### Row Padding

Each row is packed independently and padded with zeros to a multiple of 5 trits:

```
bytes_per_row = ceil(in_features / 5)
```

### Weight Reconstruction

A reconstructed weight is computed as:

```
W[i, j] = trit[i, j] × scale[group(i, j)]
```

where `group(i, j)` identifies the group containing element `(i, j)`.

## Scales

Scales control the magnitude of ternary weights:

- **group_size = 0**: One scale per tensor (shape: `[1]`)
- **group_size > 0**: One scale per group along the input dimension (shape: `[out_features × num_groups]`)

The number of groups per row:

```
num_groups = ceil(in_features / group_size)
```

## BF16 Representation

BF16 (bfloat16) is stored as a 16-bit unsigned integer (the upper 16 bits of a float32). When writing, convert:

```python
bf16_bytes = (float32_array.view(np.uint32) >> 16).astype(np.uint16)
```

When reading:

```python
float32_array = (bf16_view.astype(np.uint32) << 16).view(np.float32)
```

## Versioning

- **Major version**: Breaking changes. The engine rejects unknown major versions.
- **Minor version**: Backward-compatible additions. The engine loads older minor versions.

Current version is `0.1`.

## Metadata Keys

The following metadata keys are recognized:

| Key | Type | Description |
|-----|------|-------------|
| `source.format` | string | Input format: "safetensors" or "gguf" |
| `source.files` | array of strings | Paths to source files |
| `source.sha256` | array of strings | SHA256 hashes of source files |
| `tool.version` | string | Converter tool version |
| `hf_config` | object | Hugging Face model config.json |
| `tokenizer_json` | string | Tokenizer JSON content |
| `gguf` | object | GGUF key-value metadata |

## Example File

```python
# Write a .tern file (Python example)
import numpy as np
from tern_convert import write_tern_file

# Regular tensor
tensors = {
    "embed_tokens": np.random.randn(32000, 4096).astype(np.float32),
}
tensor_dtypes = {"embed_tokens": "f32"}

# Ternary tensor (q_proj)
from tern_convert.packing import pack_ternary_matrix
trits = np.random.randint(-1, 2, size=(4096, 4096), dtype=np.int8)
packed, _ = pack_ternary_matrix(trits)
scales = np.random.uniform(0.5, 2.0, 4096).astype(np.float32)

ternary_tensors = {
    "q_proj": {
        "packed": packed.tobytes(),
        "scales": scales,
        "group_size": 0,
        "shape": (4096, 4096),
    }
}

metadata = {
    "source_format": "safetensors",
    "tool_version": "0.1.0",
}

write_tern_file(
    "model.tern",
    tensors, tensor_dtypes,
    ternary_tensors=ternary_tensors,
    metadata=metadata,
)
```

## Reader Implementation Notes

The Tern inference engine should implement:

1. Read magic bytes `TERN` (4 bytes)
2. Read version (major in bytes 4-7, minor in bytes 8-11)
3. Reject unknown major version
4. Read header length (8 bytes, u64 LE) at bytes 12-19
5. Read JSON header (header_len bytes, padded to 64-byte boundary)
6. Parse tensor entries from JSON
7. Copy regular tensors as-is (respecting dtype conversion if needed)
8. For `tern5` tensors:
   - Unpack trits from packed bytes
   - Apply per-group scales
   - Reconstruct float32 weights