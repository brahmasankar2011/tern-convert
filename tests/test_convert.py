"""Tests for conversion orchestration."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tern_convert.convert import (
    convert,
    convert_safetensors,
    convert_gguf,
    detect_format,
    should_convert_to_ternary,
    DEFAULT_TERNARY_REGEX,
    DEFAULT_KEEP_REGEX,
)
from src.tern_convert.readers.safetensors import write_safetensors_file
from src.tern_convert.readers.gguf import write_gguf_file
from src.tern_convert.tern_format import read_tern_file, verify_tern_file


class TestShouldConvertToTernary:
    def test_hf_attention_weights(self):
        """Test HF attention weight names."""
        assert should_convert_to_ternary("model.layers.0.self_attn.q_proj.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("model.layers.0.self_attn.k_proj.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("model.layers.0.self_attn.v_proj.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("model.layers.0.self_attn.o_proj.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_hf_mlp_weights(self):
        """Test HF MLP weight names."""
        assert should_convert_to_ternary("model.layers.0.mlp.gate_proj.weight", (11008, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("model.layers.0.mlp.up_proj.weight", (11008, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("model.layers.0.mlp.down_proj.weight", (4096, 11008), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_gguf_names(self):
        """Test GGUF tensor names."""
        assert should_convert_to_ternary("blk.0.attn_q.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("blk.0.attn_k.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("blk.0.attn_v.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("blk.0.attn_output.weight", (4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("blk.0.ffn_gate.weight", (11008, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("blk.0.ffn_up.weight", (11008, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert should_convert_to_ternary("blk.0.ffn_down.weight", (4096, 11008), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_embeddings_not_ternary(self):
        """Test that embeddings are not ternary."""
        assert not should_convert_to_ternary("model.embed_tokens.weight", (32000, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert not should_convert_to_ternary("token_embd.weight", (32000, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_norms_not_ternary(self):
        """Test that norm weights are not ternary."""
        assert not should_convert_to_ternary("model.layers.0.input_layernorm.weight", (4096,), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert not should_convert_to_ternary("model.norm.weight", (4096,), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_output_head_not_ternary(self):
        """Test that output head is not ternary."""
        assert not should_convert_to_ternary("lm_head.weight", (32000, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)
        assert not should_convert_to_ternary("output.weight", (32000, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_1d_not_ternary(self):
        """Test that 1D tensors are not ternary."""
        assert not should_convert_to_ternary("q_proj.weight", (4096,), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_3d_not_ternary(self):
        """Test that 3D tensors are not ternary."""
        assert not should_convert_to_ternary("q_proj.weight", (1, 4096, 4096), DEFAULT_TERNARY_REGEX, DEFAULT_KEEP_REGEX)

    def test_keep_regex_excludes(self):
        """Test that keep_regex excludes tensors from ternary."""
        custom_keep = r".*gate_proj.*"
        assert not should_convert_to_ternary("model.layers.0.mlp.gate_proj.weight", (11008, 4096), DEFAULT_TERNARY_REGEX, custom_keep)
        # But others still match
        assert should_convert_to_ternary("model.layers.0.mlp.up_proj.weight", (11008, 4096), DEFAULT_TERNARY_REGEX, custom_keep)


class TestDetectFormat:
    def test_detect_safetensors_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            path.write_bytes(b"dummy")
            assert detect_format(path) == "safetensors"

    def test_detect_gguf_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.gguf"
            path.write_bytes(b"dummy")
            assert detect_format(path) == "gguf"

    def test_detect_safetensors_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "model-00001.safetensors").write_bytes(b"dummy")
            assert detect_format(path) == "safetensors"

    def test_detect_unknown_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.bin"
            path.write_bytes(b"dummy")
            with pytest.raises(ValueError):
                detect_format(path)


class TestConvertSafetensors:
    def test_convert_exact_mode_ternary(self):
        """Test exact mode with already-ternary weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"
            report_path = Path(tmpdir) / "report.json"

            # Create ternary weights (exactly -1, 0, 1)
            q_proj = np.array([
                [1.0, 0.0, -1.0, 1.0],
                [0.0, -1.0, 1.0, 0.0],
            ], dtype=np.float32)
            k_proj = np.array([
                [-1.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0, 1.0],
            ], dtype=np.float32)
            # Non-ternary bias
            bias = np.array([0.1, 0.2], dtype=np.float32)

            write_safetensors_file(input_path, {
                "q_proj.weight": q_proj,
                "k_proj.weight": k_proj,
                "bias": bias,
            })

            report = convert_safetensors(
                input_path, output_path,
                mode="exact", group_size=0,
                ternary_regex=DEFAULT_TERNARY_REGEX,
                keep_regex=DEFAULT_KEEP_REGEX,
                report_path=report_path,
            )

            assert report.mode == "exact"
            assert report.summary["ternary_tensors"] == 2
            assert report.summary["regular_tensors"] == 1

            # Verify report
            assert report_path.exists()
            report_data = json.loads(report_path.read_text())
            assert report_data["summary"]["ternary_tensors"] == 2

            # Verify output file
            verify_result = verify_tern_file(output_path)
            assert verify_result["valid"]

    def test_convert_exact_mode_not_ternary_fails(self):
        """Test exact mode fails on non-ternary weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            # Non-ternary weights
            q_proj = np.array([
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 1.5, 2.5, 3.5],
            ], dtype=np.float32)

            write_safetensors_file(input_path, {"q_proj.weight": q_proj})

            with pytest.raises(ValueError, match="not ternary"):
                convert_safetensors(
                    input_path, output_path,
                    mode="exact", group_size=0,
                    ternary_regex=DEFAULT_TERNARY_REGEX,
                    keep_regex=DEFAULT_KEEP_REGEX,
                )

    def test_convert_quantize_mode(self):
        """Test quantize mode with regular weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"
            report_path = Path(tmpdir) / "report.json"

            q_proj = np.array([
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0],
            ], dtype=np.float32)
            bias = np.array([0.1, 0.2], dtype=np.float32)

            write_safetensors_file(input_path, {
                "q_proj.weight": q_proj,
                "bias": bias,
            })

            report = convert_safetensors(
                input_path, output_path,
                mode="quantize", group_size=0,
                ternary_regex=DEFAULT_TERNARY_REGEX,
                keep_regex=DEFAULT_KEEP_REGEX,
                report_path=report_path,
            )

            assert report.mode == "quantize"
            assert report.summary["ternary_tensors"] == 1
            assert report.summary["regular_tensors"] == 1
            assert report.summary["avg_rel_error"] > 0

            # Verify output
            verify_result = verify_tern_file(output_path)
            assert verify_result["valid"]

    def test_convert_quantize_mode_with_groups(self):
        """Test quantize mode with group_size > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            # Wide matrix for grouping
            gate_proj = np.arange(48, dtype=np.float32).reshape(3, 16) - 24

            write_safetensors_file(input_path, {"gate_proj.weight": gate_proj})

            report = convert_safetensors(
                input_path, output_path,
                mode="quantize", group_size=8,
                ternary_regex=DEFAULT_TERNARY_REGEX,
                keep_regex=DEFAULT_KEEP_REGEX,
            )

            assert report.summary["ternary_tensors"] == 1
            # Check group size in report
            ternary_tensors = [t for t in report.tensors if t["is_ternary"]]
            assert ternary_tensors[0]["group_size"] == 8

    def test_convert_with_max_rel_error(self):
        """Test max_rel_error constraint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            q_proj = np.arange(20, dtype=np.float32).reshape(2, 10) - 10

            write_safetensors_file(input_path, {"q_proj.weight": q_proj})

            # Should fail if max_rel_error too low
            with pytest.raises(ValueError, match="exceeds max_rel_error"):
                convert_safetensors(
                    input_path, output_path,
                    mode="quantize", group_size=0,
                    ternary_regex=DEFAULT_TERNARY_REGEX,
                    keep_regex=DEFAULT_KEEP_REGEX,
                    max_rel_error=0.01,  # Very strict
                )

    def test_convert_with_custom_regex(self):
        """Test custom ternary/keep regex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            q_proj = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            k_proj = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

            write_safetensors_file(input_path, {
                "q_proj.weight": q_proj,
                "k_proj.weight": k_proj,
            })

            # Only convert q_proj
            custom_ternary = r"q_proj\.weight$"
            report = convert_safetensors(
                input_path, output_path,
                mode="quantize", group_size=0,
                ternary_regex=custom_ternary,
                keep_regex=DEFAULT_KEEP_REGEX,
            )

            assert report.summary["ternary_tensors"] == 1
            assert report.summary["regular_tensors"] == 1
            ternary_names = [t["name"] for t in report.tensors if t["is_ternary"]]
            assert ternary_names == ["q_proj.weight"]


class TestConvertGGUF:
    def test_convert_gguf_quantize(self):
        """Test GGUF conversion in quantize mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.gguf"
            output_path = Path(tmpdir) / "model.tern"

            attn_q = np.arange(20, dtype=np.float32).reshape(2, 10) - 10
            bias = np.array([0.1, 0.2], dtype=np.float32)

            write_gguf_file(input_path, {
                "blk.0.attn_q.weight": attn_q,
                "bias": bias,
            })

            report = convert_gguf(
                input_path, output_path,
                mode="quantize", group_size=0,
                ternary_regex=DEFAULT_TERNARY_REGEX,
                keep_regex=DEFAULT_KEEP_REGEX,
            )

            assert report.source_format == "gguf"
            assert report.summary["ternary_tensors"] == 1
            assert report.summary["regular_tensors"] == 1

            verify_result = verify_tern_file(output_path)
            assert verify_result["valid"]

    def test_convert_gguf_dimension_reversal(self):
        """Test that GGUF dimension reversal works in conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.gguf"
            output_path = Path(tmpdir) / "model.tern"

            # Create a tensor with known shape
            # In numpy: (out=4, in=8)
            # In GGUF: dims=[8, 4]
            attn_q = np.arange(32, dtype=np.float32).reshape(4, 8)

            write_gguf_file(input_path, {"blk.0.attn_q.weight": attn_q})

            report = convert_gguf(
                input_path, output_path,
                mode="quantize", group_size=0,
                ternary_regex=DEFAULT_TERNARY_REGEX,
                keep_regex=DEFAULT_KEEP_REGEX,
            )

            # Check shape in report
            ternary_tensors = [t for t in report.tensors if t["is_ternary"]]
            assert ternary_tensors[0]["shape"] == [4, 8]


class TestMainConvert:
    def test_main_convert_safetensors(self):
        """Test main convert entry point with safetensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            q_proj = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            write_safetensors_file(input_path, {"q_proj.weight": q_proj})

            report = convert(input_path, output_path, mode="exact")
            assert report.source_format == "safetensors"

    def test_main_convert_gguf(self):
        """Test main convert entry point with GGUF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.gguf"
            output_path = Path(tmpdir) / "model.tern"

            attn_q = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            write_gguf_file(input_path, {"blk.0.attn_q.weight": attn_q})

            report = convert(input_path, output_path, mode="exact")
            assert report.source_format == "gguf"