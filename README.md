<div align="center">

# 🔁 tern-convert

**Convert GGUF and safetensors models into the `.tern` ternary format for the Tern inference engine.**
Lossless where the math allows it, explicit about it where it doesn't.

![status](https://img.shields.io/badge/status-pre--alpha-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-brightgreen)

[What it does](#-what-it-does) · [Lossless vs lossy](#-lossless-vs-lossy) · [Install](#-install) · [Usage](#-usage) · [Connect to Tern](#-connect-to-the-tern-inference-engine) · [Format](#-the-tern-format-v01-draft) · [Development](#-development)

</div>

> **Status:** pre-alpha and under active development. Sections marked 🚧 describe target behavior that may not be implemented yet.

---

## 🧭 What it does

tern-convert reads a model and writes a single `.tern` file that the [Tern inference engine](https://github.com/brahmasankar2011/tern) can load.

| Input | Support |
|---|---|
| Hugging Face **safetensors** (one file or a directory of shards) | F32, F16, BF16. Also embeds `config.json` and `tokenizer.json` if present |
| **GGUF** | F32, F16, BF16 built in. Other tensor types need the optional `gguf` extra |

It is a separate project from the engine so the engine stays small and anyone can use the converter without building Tern.

---

## ⚖️ Lossless vs lossy

Ternary means every weight is -1, 0, or +1, multiplied by a scale. Whether a conversion loses information depends on where the model comes from.

| Starting point | What conversion does | Quality |
|---|---|---|
| Tensors that are **already ternary** | Packs them exactly | **Lossless** |
| A model **trained with ternary quantization** (QAT), stored as latent full-precision weights | Applies the same rounding rule the model uses in its own forward pass | Matches the model's intended behavior |
| An **ordinary model** (Llama-style, Qwen-style, and so on) | Rounds every weight to -1, 0, or +1 | **Lossy**, often badly for small models |

tern-convert never changes weights silently:

- **Ordinary tensors** (embeddings, norms, output head, biases) keep their original bytes unchanged.
- **Ternary tensors** are converted only in the mode you choose, and every run can write a report showing the damage.

### Modes

**`--mode exact`** (lossless)
Every target tensor must already be ternary. For each group, the absolute values must be a subset of `{0, s}` for one scale `s`. If any tensor fails, conversion stops and names the tensor and how far it is from ternary. Decoding the result reproduces the source values exactly.

**`--mode quantize`** (lossy for ordinary models)
Applies the absmean rule from the BitNet b1.58 paper, per group:

```
scale = mean(|w|) + eps
q     = clip(round(w / scale), -1, 1)
```

A warning is always printed, because for an ordinary model this is plain post-training quantization.

### Report

Pass `--report report.json` to write per-tensor details: name, shape, group size, source dtype, relative error `||W - Wq||_F / ||W||_F`, and the fraction of zeros. Pass `--max-rel-error 0.5` (for example) to make the run fail if any tensor is damaged more than that.

---

## 📦 Install

> 🚧 Not on PyPI yet. Install from source.

```bash
git clone https://github.com/<your-account>/tern-convert.git
cd tern-convert
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extra, for GGUF types like Q4_K that the built-in reader doesn't decode:

```bash
pip install -e ".[gguf]"
```

**Requirements:** Python 3.10 or newer, and NumPy. Nothing else is needed for the core features.

---

## 🚀 Usage

```bash
# Lossless: the model's ternary tensors are already exactly ternary
tern-convert convert ./my-ternary-model -o model.tern --mode exact

# Lossy: round an ordinary model to ternary, with a damage report
tern-convert convert ./my-model -o model.tern --mode quantize --group-size 128 --report report.json

# Look inside a .tern, .gguf, or .safetensors file
tern-convert inspect model.tern

# Check a .tern file is well formed
tern-convert verify model.tern

# Check an exact-mode conversion matches its source
tern-convert verify model.tern --source ./my-ternary-model
```

### Options for `convert`

| Option | Meaning |
|---|---|
| `INPUT` | A `.safetensors` or `.gguf` file, or a directory of safetensors shards |
| `-o, --output` | Output `.tern` path |
| `--mode exact\|quantize` | Conversion mode (required, no silent default) |
| `--group-size N` | Scale group size along the input dimension. `0` means one scale per tensor |
| `--ternary-regex R` | Which tensors become ternary (overrides the default rule) |
| `--keep-regex R` | Tensors that must stay in their original dtype |
| `--report FILE` | Write the per-tensor JSON report |
| `--max-rel-error X` | Fail if any ternary tensor's relative error exceeds X |

### Which tensors become ternary by default

Only the 2-D weight matrices inside transformer blocks:

- Attention: q, k, v, and output projections
- MLP: gate, up, and down projections

Both naming styles are recognized:

| Style | Names |
|---|---|
| Hugging Face | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| GGUF | `blk.N.attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down` |

Embeddings, normalization layers, the output head, and biases are never made ternary by default.

<details>
<summary><b>GGUF shape note (click to expand)</b></summary>

<br>

GGUF lists dimensions with `ne[0]` as the fastest-changing dimension, which is the reverse of NumPy's order. tern-convert reverses GGUF dimensions when reading, so a GGUF tensor with dims `(in, out)` becomes a NumPy array of shape `(out, in)`. This is covered by tests.

</details>

---

## 🔌 Connect to the Tern inference engine

The converter and the engine only share one thing: the **`.tern` file** and its written spec. That keeps the two projects independent.

```
 model (GGUF / safetensors)
        │
        ▼
  tern-convert  ──►  model.tern  ──►  Tern engine  ──►  tern run / tern serve  ──►  GUI, API clients
   (this repo)       (the contract)    (main repo)
```

### Step by step

1. **Convert**
   ```bash
   tern-convert convert ./my-model -o model.tern --mode exact
   ```
2. **Verify**
   ```bash
   tern-convert verify model.tern
   ```
3. **Run it in Tern** 🚧 (the engine's loader is not built yet)
   ```bash
   tern run ./model.tern --prompt "Hello"
   tern serve ./model.tern --port 8080
   ```
4. **Use it.** Point the Tern GUI or any OpenAI-compatible client at `http://localhost:8080`.

### The contract between the two repos

| Rule | Detail |
|---|---|
| **One spec** | `spec/FORMAT.md` in this repo is the source of truth for `.tern`. The engine's `tern-format` crate implements a reader for that spec |
| **Versioning** | The file carries a major and minor version. The engine rejects unknown major versions and loads older minor versions |
| **Metadata keys** | `source.format`, `source.files` (name and sha256), `tool.version`, `hf_config` (the model's `config.json`), `tokenizer_json` (the tokenizer text), and `gguf` (the GGUF key-values) |
| **Golden files** | Small `.tern` files plus expected decoded values live in `tests/golden/`. The engine's CI should load them and compare, so the two sides can't drift apart |
| **Repacking** | The file stores weights densely (5 trits per byte). The engine repacks them at load time into whatever layout its SIMD kernels want. That layout is the engine's business and is not part of the file format |

### What the engine's loader must do

1. Read the header and check the magic bytes `TERN`.
2. Reject an unknown major version.
3. Parse the JSON header, then locate each tensor by offset.
4. Copy ordinary tensors as they are.
5. For `tern5` tensors, unpack the trits and apply the per-group f32 scales.
6. Take the architecture from `hf_config` or `gguf` metadata and the tokenizer from `tokenizer_json`.

### Using the converter from Python 🚧

```python
from tern_convert.convert import convert

convert(
    "./my-model",
    "model.tern",
    mode="exact",          # or "quantize"
    group_size=0,
    report_path="report.json",
)
```

Check the actual function signature in `src/tern_convert/convert.py`, since this snippet shows the intended API.

---

## 📐 The `.tern` format (v0.1 draft)

The full text lives in [`spec/FORMAT.md`](spec/FORMAT.md). In short:

- **Header:** magic bytes `TERN`, `u32` major, `u32` minor, `u64` header length (all little-endian), then a UTF-8 JSON header padded with spaces so the data section starts on a 64-byte boundary.
- **Data:** every tensor starts on a 64-byte boundary. Offsets in the JSON are relative to the start of the data section.
- **JSON header:** `format_version`, `metadata`, and `tensors`.

**Ordinary tensor entry**

```json
{ "dtype": "bf16", "shape": [4096, 4096], "offset": 0, "nbytes": 33554432 }
```

**Ternary tensor entry**

```json
{
  "dtype": "tern5",
  "shape": [4096, 4096],
  "group_size": 128,
  "packed": { "offset": 33554432, "nbytes": 3358720 },
  "scales": { "offset": 36913152, "nbytes": 524288, "dtype": "f32", "count": 131072 }
}
```

**Packing:** 5 trits per byte in base 3. Each trit maps to a digit (`-1 → 0`, `0 → 1`, `+1 → 2`), and a byte is `d0 + 3·d1 + 9·d2 + 27·d3 + 81·d4`, so valid bytes are 0 to 242. Each row is packed on its own and padded with zeros to a multiple of 5, so `bytes_per_row = ceil(in / 5)`. A weight is `trit × scale`. That is about 1.6 bits per weight.

**Scales:** stored as f32. Group size 0 means one scale for the whole tensor. Otherwise groups run along the input dimension.

---

## ⚠️ Limitations

- Ternary only pays off for models that behave well as ternary. Converting an ordinary model usually hurts quality, often a lot for small ones.
- GGUF quantized types (Q4_K and similar) go through the optional `gguf` package. That path is less tested than the built-in readers.
- Native GGUF ternary types (TQ1_0, TQ2_0) are not supported in v0.1.
- Check every model's license before converting or sharing the result. A converted model is generally a derivative of the original.

---

## 🛠️ Development

### Layout

```
tern-convert/
├── src/tern_convert/
│   ├── __init__.py
│   ├── packing.py        # 5-trits-per-byte pack and unpack
│   ├── tern_format.py    # .tern reader and writer
│   ├── quantize.py       # exact check and absmean quantizer
│   ├── convert.py        # the conversion pipeline and report
│   ├── cli.py            # argparse CLI (convert, inspect, verify)
│   └── readers/
│       ├── safetensors.py
│       └── gguf.py
├── tests/
│   ├── helpers/          # tiny safetensors and GGUF writers for fixtures
│   ├── golden/           # small .tern files with expected values
│   └── test_*.py
├── spec/FORMAT.md
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── THIRD_PARTY_NOTICES.md
├── LICENSE
└── .gitignore
```

### Run the tests

```bash
pip install -e ".[dev]"
pytest -q
```

### Rules for contributors

1. **No code from other inference projects.** The readers are written from the public safetensors and GGUF format descriptions. Do not copy code from bitnet.cpp, llama.cpp, or ggml.
2. **Runtime dependency is NumPy only.** Anything else is an optional extra.
3. **Never change weights silently.** Lossy steps need an explicit flag and a report.
4. **Assert, never clamp.** A value outside the expected set is an error, not something to round.
5. **Process one tensor at a time** so large models do not have to fit in memory.
6. **Sign off every commit:** `git commit -s`. No CLA.

### Test coverage that must exist

- Pack and unpack round-trip, including lengths that are not multiples of 5.
- Exact mode accepts truly ternary data and rejects data that is almost ternary.
- Quantize mode produces the expected relative error on a known tensor.
- GGUF dimension reversal.
- BF16 reading and writing.
- Header alignment and tensor offsets.
- `verify` catches a deliberately corrupted file.
- Golden files decode to their expected values.

---

## 🗺️ Roadmap

- [ ] v0.1: safetensors and GGUF readers, packing, `exact` and `quantize`, `inspect`, `verify`
- [ ] Real-model testing with published error reports
- [ ] Native GGUF ternary types (TQ1_0, TQ2_0)
- [ ] Shared conformance tests with the Tern engine's loader
- [ ] Optional training-aware conversion helpers (distillation recipe)

## 📜 License

[Apache-2.0](LICENSE). Contributions are accepted under the DCO. Models you convert keep their own licenses.

## 🙏 Acknowledgements

Built from the public safetensors and GGUF format descriptions and the published BitNet b1.58 research. Part of the [Tern](https://github.com/brahmasankar2011/tern) project.
