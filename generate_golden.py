#!/usr/bin/env python3
"""Generate golden test files for tern-convert.

This script generates all golden test files needed by test_golden.py.
Run this before running pytest on the golden tests.
"""

from pathlib import Path
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tern_convert.tern_format import write_tern_file
from tern_convert.packing import pack_ternary_matrix

GOLDEN_DIR = Path(__file__).parent / "tests" / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

# simple.tern - Basic float32 tensors
print("Generating simple.tern...")
tensors = {
    'weight': np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    'bias': np.array([0.5, -0.5], dtype=np.float32),
}
tensor_dtypes = {'weight': 'f32', 'bias': 'f32'}
metadata = {'format_version': '0.1', 'test_case': 'golden_simple'}
write_tern_file(GOLDEN_DIR / 'simple.tern', tensors, tensor_dtypes, metadata=metadata)

# ternary.tern - Ternary tensor with known values (per-tensor scale)
print("Generating ternary.tern...")
shape = (4, 10)
trits = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
                  [0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
                  [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
                  [-1, 1, 0, -1, 1, 0, -1, 1, 0, -1]], dtype=np.int8)
packed, _ = pack_ternary_matrix(trits)
# group_size=0 means ONE scale per TENSOR (not per row)
scales = np.array([1.0], dtype=np.float32)
ternary_tensors = {'q_proj': {
        'packed': packed.tobytes(),
        'scales': scales,
        'group_size': 0,
        'shape': shape,
    }}
write_tern_file(GOLDEN_DIR / 'ternary.tern', {}, {}, ternary_tensors=ternary_tensors, metadata={'format_version': '0.1', 'test_case': 'golden_ternary'})

# bf16.tern - BF16 encoded tensor
print("Generating bf16.tern...")
f32_data = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
bf16_data = (f32_data.view(np.uint32) >> 16).astype(np.uint16)
write_tern_file(GOLDEN_DIR / 'bf16.tern', {'bf_weight': bf16_data}, {'bf_weight': 'bf16'}, metadata={'format_version': '0.1', 'test_case': 'golden_bf16'})

# mixed.tern - Mixed regular and ternary tensors (per-tensor scale)
print("Generating mixed.tern...")
shape = (2, 8)
trits = np.array([[-1, 0, 1, -1, 0, 1, -1, 0],
                  [1, -1, 0, 1, -1, 0, 1, -1]], dtype=np.int8)
packed, _ = pack_ternary_matrix(trits)
# group_size=0 means ONE scale per TENSOR (not per row)
scales = np.array([1.0], dtype=np.float32)  # Single scale for entire tensor
write_tern_file(GOLDEN_DIR / 'mixed.tern', {'attention_norm': np.ones(128, dtype=np.float32)}, {'attention_norm': 'f32'}, ternary_tensors={'ffn_gate': {'packed': packed.tobytes(), 'scales': scales, 'group_size': 0, 'shape': shape}}, metadata={'format_version': '0.1', 'test_case': 'golden_mixed'})

print("All golden files generated successfully!")
for f in sorted(GOLDEN_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size} bytes")