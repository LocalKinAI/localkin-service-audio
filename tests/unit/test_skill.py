"""The agent skill (`kin skill`): installable, and true to the CLI it teaches.

The old OpenClaw skill drifted from the commands it described; these tests
fail as soon as the skill uses an option a `kin` command doesn't have.
"""
from __future__ import annotations

import re
import shlex
from importlib import resources
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from localkin_service_audio.cli.main import cli

PACKAGED = resources.files("localkin_service_audio") / "skills" / "localkin-audio" / "SKILL.md"
OPENCLAW = Path(__file__).resolve().parents[2] / "openclaw-skill" / "localkin-audio" / "SKILL.md"


def _split(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    return m.group(1), text[m.end():]


def _kin_lines(text):
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            if line.strip().startswith("kin "):
                yield shlex.split(line, comments=True)


def _command(words):
    """The click command a `kin ...` line runs, and the words after it."""
    command, rest = cli, words[1:]
    while isinstance(command, click.Group) and rest and rest[0] in command.commands:
        command, rest = command.commands[rest[0]], rest[1:]
    return command, rest


def test_frontmatter_fits_the_skill_format():
    front, _ = _split(PACKAGED.read_text(encoding="utf-8"))
    name = re.search(r"^name: (.*)$", front, re.M).group(1)
    description = re.search(r"^description: (.*)$", front, re.M).group(1)
    assert re.fullmatch(r"[a-z0-9-]{1,64}", name)
    assert 0 < len(description) <= 1024


def test_openclaw_copy_has_the_same_body():
    assert _split(OPENCLAW.read_text(encoding="utf-8"))[1] == _split(PACKAGED.read_text(encoding="utf-8"))[1]


def test_every_option_in_the_skill_exists():
    lines = list(_kin_lines(PACKAGED.read_text(encoding="utf-8")))
    assert len(lines) > 10
    unknown = []
    for words in lines:
        command, rest = _command(words)
        assert not isinstance(command, click.Group), f"incomplete command: {' '.join(words)}"
        known = {opt for p in command.params if isinstance(p, click.Option)
                 for opt in p.opts + p.secondary_opts} | {"--help", "-h"}
        unknown += [f"{' '.join(words[:3])}: {w}" for w in rest
                    if w.startswith("-") and w.split("=")[0] not in known]
    assert not unknown, unknown


def test_kin_skill_installs_and_prints(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["skill", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    installed = tmp_path / "localkin-audio" / "SKILL.md"
    assert installed.read_text(encoding="utf-8") == PACKAGED.read_text(encoding="utf-8")
    printed = runner.invoke(cli, ["skill", "--print"])
    assert printed.exit_code == 0 and printed.output.startswith("---\nname: localkin-audio")


@pytest.mark.parametrize("args", [["audio", "tts"], ["audio", "tts", "--model", "kokoro"]])
def test_tts_without_text_is_an_error_before_loading(args, monkeypatch):
    from localkin_service_audio.core.audio_processing import engine

    monkeypatch.setattr(engine.AudioEngine, "load_tts", lambda *a, **k: pytest.fail("loaded a model"))
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0 and "Text required" in result.output
