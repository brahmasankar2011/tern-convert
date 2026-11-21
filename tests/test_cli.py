"""Tests for the CLI."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tern_convert.cli import main
from src.tern_convert.readers.safetensors import write_safetensors_file


class TestConvertCommand:
    """Tests for the 'tern-convert convert' command."""

    def test_convert_exact_mode(self):
        """Test convert command with exact mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"
            report_path = Path(tmpdir) / "report.json"

            # Create ternary weights
            q_proj = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            write_safetensors_file(input_path, {"q_proj.weight": q_proj})

            result = main([
                "convert", str(input_path),
                "-o", str(output_path),
                "--mode", "exact",
                "--report", str(report_path),
            ])

            assert result == 0
            assert output_path.exists()
            assert report_path.exists()

            # Check report
            report = json.loads(report_path.read_text())
            assert report["mode"] == "exact"

    def test_convert_quantize_mode(self):
        """Test convert command with quantize mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            weights = np.random.randn(4, 8).astype(np.float32)
            write_safetensors_file(input_path, {"q_proj.weight": weights})

            result = main([
                "convert", str(input_path),
                "-o", str(output_path),
                "--mode", "quantize",
                "--group-size", "4",
            ])

            assert result == 0
            assert output_path.exists()

    def test_convert_with_max_rel_error(self):
        """Test convert with max_rel_error constraint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"
            output_path = Path(tmpdir) / "model.tern"

            weights = np.arange(20, dtype=np.float32).reshape(2, 10) - 10
            write_safetensors_file(input_path, {"q_proj.weight": weights})

            # Should fail with very strict max_rel_error
            result = main([
                "convert", str(input_path),
                "-o", str(output_path),
                "--mode", "quantize",
                "--max-rel-error", "0.01",
            ])

            assert result == 1


class TestInspectCommand:
    """Tests for the 'tern-convert inspect' command."""

    def test_inspect_tern_file(self):
        """Test inspect on a .tern file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.tern_convert.tern_format import write_tern_file

            input_path = Path(tmpdir) / "model.tern"

            tensors = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
            tensor_dtypes = {"weight": "f32"}

            write_tern_file(input_path, tensors, tensor_dtypes, metadata={"test": "value"})

            result = main(["inspect", str(input_path)])

            assert result == 0

    def test_inspect_safetensors_file(self):
        """Test inspect on a .safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.safetensors"

            tensors = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
            write_safetensors_file(input_path, tensors)

            result = main(["inspect", str(input_path)])

            assert result == 0

    def test_inspect_gguf_file(self):
        """Test inspect on a .gguf file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.tern_convert.readers.gguf import write_gguf_file

            input_path = Path(tmpdir) / "model.gguf"

            tensors = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
            write_gguf_file(input_path, tensors)

            result = main(["inspect", str(input_path)])

            assert result == 0


class TestVerifyCommand:
    """Tests for the 'tern-convert verify' command."""

    def test_verify_valid_file(self):
        """Test verify on a valid .tern file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.tern_convert.tern_format import write_tern_file

            input_path = Path(tmpdir) / "model.tern"

            tensors = {"weight": np.array([[1.0, 2.0]], dtype=np.float32)}
            tensor_dtypes = {"weight": "f32"}

            write_tern_file(input_path, tensors, tensor_dtypes)

            result = main(["verify", str(input_path)])

            assert result == 0

    def test_verify_with_source(self):
        """Test verify with source comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.tern_convert.tern_format import write_tern_file

            tern_path = Path(tmpdir) / "model.tern"
            source_path = Path(tmpdir) / "model.safetensors"

            # Create source
            weights = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            write_safetensors_file(source_path, {"q_proj.weight": weights})

            # Convert with exact mode
            main([
                "convert", str(source_path),
                "-o", str(tern_path),
                "--mode", "exact",
                "--ternary-regex", r"q_proj\.weight$",
            ])

            # Verify with source
            result = main(["verify", str(tern_path), "--source", str(source_path)])

            assert result == 0


class TestHelp:
    """Tests for help messages."""

    def test_main_help(self):
        """Test that main prints help with no arguments."""
        result = main([])
        assert result == 0

    def test_convert_help(self):
        """Test convert command help."""
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            main(["convert", "--help"])
        assert exc_info.value.code == 0