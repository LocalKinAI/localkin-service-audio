"""
Tests for CLI commands using Click's CliRunner.
"""
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from localkin_service_audio.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


class TestCLIBasic:
    """Tests for basic CLI functionality."""

    def test_cli_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "kin" in result.output.lower() or "version" in result.output.lower()

    def test_cli_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "LocalKin Audio" in result.output or "audio" in result.output

    def test_audio_group_help(self, runner):
        """Test audio group help."""
        result = runner.invoke(cli, ["audio", "--help"])

        assert result.exit_code == 0
        assert "transcribe" in result.output
        assert "tts" in result.output
        assert "models" in result.output


class TestModelsCommand:
    """Tests for the models command."""

    def test_models_list(self, runner):
        """Test listing models."""
        result = runner.invoke(cli, ["audio", "models"])

        # Should show model list or headers
        assert result.exit_code == 0

    def test_models_list_stt(self, runner):
        """Test listing only STT models."""
        result = runner.invoke(cli, ["audio", "models", "--type", "stt"])

        assert result.exit_code == 0

    def test_models_list_tts(self, runner):
        """Test listing only TTS models."""
        result = runner.invoke(cli, ["audio", "models", "--type", "tts"])

        assert result.exit_code == 0


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_basic(self, runner):
        """Test basic info command."""
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Version" in result.output or "version" in result.output.lower()

    def test_info_verbose(self, runner):
        """Test verbose info command."""
        result = runner.invoke(cli, ["info", "--verbose"])

        assert result.exit_code == 0


class TestRecommendCommand:
    """Tests for the recommend command."""

    def test_recommend(self, runner):
        """Test model recommendation."""
        result = runner.invoke(cli, ["audio", "recommend"])

        # May fail if no audio hardware, but should not crash
        assert result.exit_code in [0, 1]


class TestTranscribeCommand:
    """Tests for the transcribe command."""

    def test_transcribe_no_file(self, runner):
        """Test transcribe without file shows error."""
        result = runner.invoke(cli, ["audio", "transcribe"])

        # Should show usage error (missing required argument)
        assert result.exit_code != 0

    def test_transcribe_help(self, runner):
        """Test transcribe help."""
        result = runner.invoke(cli, ["audio", "transcribe", "--help"])

        assert result.exit_code == 0
        assert "AUDIO_PATH" in result.output or "audio" in result.output.lower()

    def test_transcribe_with_mock(self, runner, sample_audio_file):
        """Test transcribe with mocked engine."""
        from localkin_service_audio.core.types import TranscriptionResult

        with patch("localkin_service_audio.core.get_audio_engine") as mock_engine_fn:
            mock_engine = MagicMock()
            mock_engine._stt_strategy = None
            mock_engine._stt_model_name = None
            mock_engine.load_stt.return_value = True
            mock_engine.transcribe.return_value = TranscriptionResult(
                text="Hello world",
                language="en",
            )
            mock_engine_fn.return_value = mock_engine

            result = runner.invoke(
                cli, ["audio", "transcribe", sample_audio_file, "--model", "mock:base"]
            )

            # Should call the engine methods
            mock_engine.load_stt.assert_called_once()
            mock_engine.transcribe.assert_called_once()


class TestTTSCommand:
    """Tests for the TTS command."""

    def test_tts_no_text(self, runner):
        """Test TTS without text shows error."""
        result = runner.invoke(cli, ["audio", "tts"])

        # Should show usage error (missing required argument)
        assert result.exit_code != 0

    def test_tts_help(self, runner):
        """Test TTS help."""
        result = runner.invoke(cli, ["audio", "tts", "--help"])

        assert result.exit_code == 0
        assert "TEXT" in result.output or "text" in result.output.lower()


class TestPullCommand:
    """Tests for the pull command."""

    def test_pull_help(self, runner):
        """Test pull help."""
        result = runner.invoke(cli, ["audio", "pull", "--help"])

        assert result.exit_code == 0
        assert "MODEL" in result.output or "model" in result.output.lower()

    def test_pull_no_model(self, runner):
        """Test pull without model shows error."""
        result = runner.invoke(cli, ["audio", "pull"])

        # Should show usage error
        assert result.exit_code != 0


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_help(self, runner):
        """Test serve help."""
        result = runner.invoke(cli, ["audio", "serve", "--help"])

        assert result.exit_code == 0
        assert "port" in result.output.lower() or "host" in result.output.lower()


class TestMCPCommand:
    """Tests for the MCP command."""

    def test_mcp_help(self, runner):
        """Test MCP help."""
        result = runner.invoke(cli, ["mcp", "--help"])

        assert result.exit_code == 0


class TestWebCommand:
    """Tests for the web command."""

    def test_web_help(self, runner):
        """Test web help."""
        result = runner.invoke(cli, ["web", "--help"])

        assert result.exit_code == 0


class TestBenchmarkCommand:
    """Tests for the benchmark command."""

    def test_benchmark_help(self, runner):
        """Test benchmark help."""
        result = runner.invoke(cli, ["audio", "benchmark", "--help"])

        assert result.exit_code == 0


class TestListenCommand:
    """Tests for the listen command."""

    def test_listen_help(self, runner):
        """Test listen help."""
        result = runner.invoke(cli, ["audio", "listen", "--help"])

        assert result.exit_code == 0
        assert "Real-time" in result.output or "listen" in result.output.lower()
        assert "--tts" in result.output
        assert "--llm" in result.output


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_help(self, runner):
        """Test config help."""
        result = runner.invoke(cli, ["audio", "config", "--help"])

        assert result.exit_code == 0
        assert "--path" in result.output
        assert "--models" in result.output
        assert "--init" in result.output

    def test_config_default(self, runner):
        """Test config with no options."""
        result = runner.invoke(cli, ["audio", "config"])

        assert result.exit_code == 0
        assert "Config" in result.output or "config" in result.output.lower()

    def test_config_path(self, runner):
        """Test config --path."""
        result = runner.invoke(cli, ["audio", "config", "--path"])

        assert result.exit_code == 0
        assert "localkin" in result.output.lower()

    def test_config_models(self, runner):
        """Test config --models."""
        result = runner.invoke(cli, ["audio", "config", "--models"])

        assert result.exit_code == 0


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_help(self, runner):
        """Test status help."""
        result = runner.invoke(cli, ["audio", "status", "--help"])

        assert result.exit_code == 0
        assert "system" in result.output.lower() or "status" in result.output.lower()

    def test_status_runs(self, runner):
        """Test status command runs and shows library checks."""
        result = runner.invoke(cli, ["audio", "status"])

        assert result.exit_code == 0
        assert "STT Libraries" in result.output
        assert "TTS Libraries" in result.output
        assert "ML Libraries" in result.output
        assert "Model Registry" in result.output
        assert "Configuration" in result.output
        assert "Cache" in result.output


class TestCacheCommand:
    """Tests for the cache command."""

    def test_cache_help(self, runner):
        """Test cache help."""
        result = runner.invoke(cli, ["audio", "cache", "--help"])

        assert result.exit_code == 0
        assert "info" in result.output
        assert "clear" in result.output

    def test_cache_info(self, runner):
        """Test cache info command."""
        result = runner.invoke(cli, ["audio", "cache", "info"])

        assert result.exit_code == 0
        assert "Cache" in result.output

    def test_cache_default_shows_info(self, runner):
        """Test that bare cache command shows info."""
        result = runner.invoke(cli, ["audio", "cache"])

        assert result.exit_code == 0
        assert "Cache" in result.output

    def test_cache_clear_abort(self, runner):
        """Test cache clear aborts when user says no."""
        result = runner.invoke(cli, ["audio", "cache", "clear"], input="n\n")

        # Should abort (exit code 1 from click.Abort)
        assert result.exit_code != 0 or "Aborted" in result.output


class TestPSCommand:
    """Tests for the ps command."""

    def test_ps_help(self, runner):
        """Test ps help."""
        result = runner.invoke(cli, ["audio", "ps", "--help"])

        assert result.exit_code == 0
        assert "running" in result.output.lower() or "server" in result.output.lower()

    def test_ps_runs(self, runner):
        """Test ps command runs."""
        result = runner.invoke(cli, ["audio", "ps"])

        assert result.exit_code == 0
        assert "Scanning" in result.output or "server" in result.output.lower()


class TestAddModelCommand:
    """Tests for the add-model command."""

    def test_add_model_help(self, runner):
        """Test add-model help."""
        result = runner.invoke(cli, ["audio", "add-model", "--help"])

        assert result.exit_code == 0
        assert "--template" in result.output
        assert "--repo" in result.output
        assert "--name" in result.output

    def test_add_model_requires_template_or_repo(self, runner):
        """Test add-model requires --template or --repo."""
        result = runner.invoke(cli, ["audio", "add-model", "--name", "test"])

        assert result.exit_code != 0

    def test_add_model_requires_name(self, runner):
        """Test add-model requires --name."""
        result = runner.invoke(cli, ["audio", "add-model", "--template", "whisper_stt"])

        assert result.exit_code != 0

    def test_add_model_invalid_template(self, runner):
        """Test add-model with invalid template."""
        result = runner.invoke(
            cli, ["audio", "add-model", "--template", "nonexistent", "--name", "test"]
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_add_model_with_template(self, runner):
        """Test add-model with valid template."""
        with patch("localkin_service_audio.core.config_legacy.save_models_config", return_value=True):
            result = runner.invoke(
                cli,
                ["audio", "add-model", "--template", "whisper_stt", "--name", "test-whisper"],
                input="y\n",  # Confirm overwrite if already exists
            )

        # Should succeed or prompt for overwrite
        assert "test-whisper" in result.output


class TestListTemplatesCommand:
    """Tests for the list-templates command."""

    def test_list_templates_help(self, runner):
        """Test list-templates help."""
        result = runner.invoke(cli, ["audio", "list-templates", "--help"])

        assert result.exit_code == 0

    def test_list_templates_runs(self, runner):
        """Test list-templates shows templates."""
        result = runner.invoke(cli, ["audio", "list-templates"])

        assert result.exit_code == 0
        assert "Templates" in result.output
        assert "whisper_stt" in result.output
        assert "bark_tts" in result.output
        assert "Usage" in result.output


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_command(self, runner):
        """Test invalid command shows error."""
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code != 0

    def test_invalid_audio_subcommand(self, runner):
        """Test invalid audio subcommand shows error."""
        result = runner.invoke(cli, ["audio", "invalid-subcommand"])

        assert result.exit_code != 0

    def test_transcribe_nonexistent_file(self, runner):
        """Test transcribe with nonexistent file."""
        result = runner.invoke(cli, ["audio", "transcribe", "/nonexistent/file.wav"])

        # Should fail with file not found or model loading error
        assert result.exit_code != 0
