"""
JSON-lines protocol between the server and an isolated worker.

Requests arrive one per line on stdin; each gets exactly one reply line:
``{"ok": true, ...}`` or ``{"ok": false, "error": "..."}``. Audio moves
through WAV files named in the messages rather than inline.

Model libraries print freely, so the reply channel is a private copy of the
original stdout, and both ``sys.stdout`` and file descriptor 1 are pointed at
stderr before any handler runs: neither a stray print, nor native code, nor a
child process writing to stdout can corrupt the protocol.
"""
import json
import os
import sys
import traceback


def serve(handlers):
    reply = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), 1)
    sys.stdout = sys.stderr
    reply.write(json.dumps({"ok": True, "ready": True}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.pop("op")
            if op == "shutdown":
                reply.write(json.dumps({"ok": True}) + "\n")
                return
            result = handlers[op](**request)
            reply.write(json.dumps({"ok": True, **(result or {})}) + "\n")
        except Exception as e:
            traceback.print_exc()
            reply.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) + "\n")


def write_wav(path, audio, sample_rate):
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    sf.write(path, np.squeeze(audio).astype(np.float32), int(sample_rate))
    return {"path": path, "sample_rate": int(sample_rate)}


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
