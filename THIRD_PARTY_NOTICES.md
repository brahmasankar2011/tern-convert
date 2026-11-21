# Third-Party Notices

tern-convert includes software from the following third-party projects:

## safetensors Format Specification

- **Source**: https://github.com/huggingface/safetensors
- **License**: Apache-2.0
- **Usage**: The reader and writer are implemented from the public format specification, not derived from the reference code.

## GGUF Format Specification

- **Source**: https://github.com/ggerganov/ggml
- **License**: MIT OR Apache-2.0
- **Usage**: The reader is implemented from the public format specification at https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

## BitNet b1.58 Paper

- **Source**: BitNet b1.58: Sign-only Quantization
- **Authors**: Ze Liu, Yixuan Du, et al.
- **Usage**: The absmean quantization formula is based on Section 4.1 of the paper

## NumPy

- **Source**: https://numpy.org/
- **License**: NumPy License (BSD-3-Clause)
- **Usage**: Required dependency for array operations

## Python Standard Library

- Modules used: `struct`, `json`, `hashlib`, `argparse`, `dataclasses`, `pathlib`, `typing`
- License: Python Software License

## pytest

- **Source**: https://docs.pytest.org/
- **License**: MIT
- **Usage**: Testing framework (dev dependency only)

---

For more information about licenses, see:
- tern-convert: Apache-2.0 (see LICENSE file)
- Third-party components: individual license terms as listed above