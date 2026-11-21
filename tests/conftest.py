"""Pytest configuration for tern-convert tests.

This module sets up fixtures and ensures golden test files exist.
"""

from pathlib import Path

import numpy as np
import pytest

from src.tern_convert.tern_format import write_tern_file
from src.tern_convert.packing import pack_ternary_matrix


GOLDEN_DIR = Path(__file__).parent / "golden"


def _ensure_golden_files():
    """Generate golden test files if they don't exist."""
    GOLDEN_DIR.mkdir(exist_ok=True)

    # simple.tern - Basic float32 tensors
    simple_path = GOLDEN_DIR / "simple.tern"
    if not simple_path.exists():
        tensors = {
            'weight': np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            'bias': np.array([0.5, -0.5], dtype=np.float32),
        }
        tensor_dtypes = {'weight': 'f32', 'bias': 'f32'}
        metadata = {'format_version': '0.1', 'test_case': 'golden_simple'}
        write_tern_file(simple_path, tensors, tensor_dtypes, metadata=metadata)

    # ternary.tern - Ternary tensor with known values
    ternary_path = GOLDEN_DIR / "ternary.tern"
    if not ternary_path.exists():
        shape = (4, 10)
        trits = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
                          [0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
                          [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
                          [-1, 1, 0, -1, 1, 0, -1, 1, 0, -1]], dtype=np.int8)
        packed, _ = pack_ternary_matrix(trits)
        scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        ternary_tensors = {
            'q_proj': {
                'packed': packed.tobytes(),
                'scales': scales,
                'group_size': 0,
                'shape': shape,
            }
        }

        metadata = {'format_version': '0.1', 'test_case': 'golden_ternary', 'trit_shape': str(list(shape))}
        write_tern_file(ternary_path, {}, {}, ternary_tensors=ternary_tensors, metadata=metadata)

    # bf16.tern - BF16 encoded tensor
    bf16_path = GOLDEN_DIR / "bf16.tern"
    if not bf16_path.exists():
        f32_data = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
        bf16_data = (f32_data.view(np.uint32) >> 16).astype(np.uint16)

        tensors = {'bf_weight': bf16_data}
        tensor_dtypes = {'bf_weight': 'bf16'}
        metadata = {'format_version': '0.1', 'test_case': 'golden_bf16'}

        write_tern_file(bf16_path, tensors, tensor_dtypes, metadata=metadata)

    # mixed.tern - Mixed regular and ternary tensors
    mixed_path = GOLDEN_DIR / "mixed.tern"
    if not mixed_path.exists():
        shape = (2, 8)
        trits = np.array([[-1, 0, 1, -1, 0, 1, -1, 0],
                          [1, -1, 0, 1, -1, 0, 1, -1]], dtype=np.int8)
        packed, _ = pack_ternary_matrix(trits)
        scales = np.array([0.5, 2.0], dtype=np.float32)

        ternary_tensors = {
            'ffn_gate': {
                'packed': packed.tobytes(),
                'scales': scales,
                'group_size': 0,
                'shape': shape,
            }
        }

        tensors = {'attention_norm': np.ones(128, dtype=np.float32)}
        tensor_dtypes = {'attention_norm': 'f32'}
        metadata = {'format_version': '0.1', 'test_case': 'golden_mixed'}

        write_tern_file(mixed_path, tensors, tensor_dtypes,
                        ternary_tensors=ternary_tensors, metadata=metadata)


@pytest.fixture(scope="session", autouse=True)
def ensure_golden_files():
    """Ensure golden test files exist before running tests."""
    _ensure_golden_files()