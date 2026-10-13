"""Text files are read and written as UTF-8 everywhere in the package.

Without an encoding, open() and Path.read_text/write_text use the locale's:
cp1252 on Windows, which can't write Chinese and garbles reading it. The
first Windows CI run failed on exactly that.
"""
import ast
from pathlib import Path

import localkin_service_audio

ROOT = Path(localkin_service_audio.__file__).parent
# Modules whose open() isn't a text file: os.open, soundfile, wave, PIL...
NOT_TEXT = {"os", "sf", "soundfile", "wave", "tarfile", "zipfile", "gzip", "Image", "webbrowser"}


def _binary_mode(call: ast.Call) -> bool:
    modes = [a.value for a in call.args[:2] if isinstance(a, ast.Constant)]
    modes += [k.value.value for k in call.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
    return any(isinstance(m, str) and set(m) <= set("rwxabt+") and "b" in m for m in modes)


def _text_io_without_encoding(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        owner = getattr(getattr(func, "value", None), "id", None)
        if name in ("read_text", "write_text"):
            yield node.lineno
        elif name == "open" and owner not in NOT_TEXT and not _binary_mode(node):
            yield node.lineno


def test_text_files_name_their_encoding():
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue  # e.g. a virtualenv left inside a checkout
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{rel}:{line}" for line in _text_io_without_encoding(tree)]
    assert not found, "text file I/O without encoding=: " + ", ".join(found)


def test_the_check_catches_what_it_should():
    tree = ast.parse(
        "open(p)\nopen(p, 'w')\np.read_text()\np.write_text(s)\n"
        "open(p, 'rb')\nopen(p, encoding='utf-8')\nos.open(p, 0)\nsf.open(p)\np.open('rb')\n"
    )
    assert sorted(_text_io_without_encoding(tree)) == [1, 2, 3, 4]
