"""Tests for CLI help and version commands."""

from click.testing import CliRunner

from llm_compress import __version__
from llm_compress.cli import main


class TestCliHelp:
    """Test CLI help commands."""

    def test_main_help(self) -> None:
        """Test that --help shows all commands and options."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "LLM Compress: Unified LLM Quantization & Inference System" in result.output
        assert "Commands:" in result.output
        assert "download" in result.output
        assert "list" in result.output
        assert "quantize" in result.output
        assert "remove" in result.output
        assert "serve" in result.output
        assert "tui" in result.output
        assert "--version" in result.output
        assert "--help" in result.output

    def test_download_help(self) -> None:
        """Test download subcommand --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])

        assert result.exit_code == 0
        assert "Download a model from HuggingFace Hub" in result.output
        assert "MODEL_ID" in result.output
        assert "--cache-dir" in result.output
        assert "--token" in result.output

    def test_quantize_help(self) -> None:
        """Test quantize subcommand --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["quantize", "--help"])

        assert result.exit_code == 0
        assert "Quantize a downloaded model" in result.output
        assert "MODEL_ID" in result.output
        assert "--bits" in result.output
        assert "--kv-cache" in result.output
        assert "--cache-dir" in result.output

    def test_serve_help(self) -> None:
        """Test serve subcommand --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])

        assert result.exit_code == 0
        assert "Start the OpenAI-compatible API server" in result.output
        assert "MODEL_ID" in result.output
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--backend" in result.output
        assert "--kv-cache" in result.output
        assert "--cache-dir" in result.output
        assert "vllm" in result.output
        assert "llama-cpp" in result.output

    def test_list_help(self) -> None:
        """Test list subcommand --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["list", "--help"])

        assert result.exit_code == 0
        assert "List all downloaded models" in result.output
        assert "--cache-dir" in result.output

    def test_remove_help(self) -> None:
        """Test remove subcommand --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["remove", "--help"])

        assert result.exit_code == 0
        assert "Remove a downloaded model" in result.output
        assert "MODEL_ID" in result.output
        assert "--cache-dir" in result.output
        assert "--force" in result.output

    def test_tui_help(self) -> None:
        """Test tui subcommand --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["tui", "--help"])

        assert result.exit_code == 0
        assert "Launch the terminal user interface" in result.output


class TestCliVersion:
    """Test CLI version command."""

    def test_version_flag(self) -> None:
        """Test that --version shows the semantic version."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "llm-compress, version" in result.output
        assert __version__ in result.output

    def test_version_format(self) -> None:
        """Test that version follows semantic versioning format."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        # Check version format (major.minor.patch or major.minor.patch-dev)
        version_parts = __version__.split(".")
        assert len(version_parts) >= 2
        assert version_parts[0].isdigit()  # major
        assert version_parts[1].isdigit()  # minor


class TestHelpTextQuality:
    """Test that help text is clear and comprehensive."""

    def test_main_help_has_examples(self) -> None:
        """Test that main help includes usage examples."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Examples:" in result.output
        assert "download" in result.output
        assert "quantize" in result.output
        assert "serve" in result.output

    def test_serve_help_comprehensive(self) -> None:
        """Test that serve help is comprehensive with endpoint info."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])

        assert result.exit_code == 0
        # Should mention health check
        assert "health" in result.output.lower()
        # Should mention models endpoint
        assert "models" in result.output.lower()
        # Should mention chat completions
        assert "chat" in result.output.lower()

    def test_quantize_help_has_description(self) -> None:
        """Test that quantize help has clear description."""
        runner = CliRunner()
        result = runner.invoke(main, ["quantize", "--help"])

        assert result.exit_code == 0
        # Should describe what quantization does
        assert "quantizes" in result.output.lower() or "quantize" in result.output
        # Should mention bit width options
        assert "4" in result.output
        assert "8" in result.output

    def test_download_help_has_model_id_description(self) -> None:
        """Test that download help explains MODEL_ID format."""
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])

        assert result.exit_code == 0
        # Should explain MODEL_ID format (e.g., meta-llama/Llama-2-7b-hf)
        assert "HuggingFace" in result.output
        assert "identifier" in result.output or "MODEL_ID" in result.output
