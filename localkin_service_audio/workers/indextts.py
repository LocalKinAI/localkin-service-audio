"""IndexTTS-2 / 2.5 (bilibili), installed from git; always clones a reference."""
import os

from _protocol import serve, write_wav

state = {}


def load(repo, params=None):
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(repo)
    if "2.5" in repo:
        from indextts.infer_v2_5 import IndexTTS2
        state["model"] = IndexTTS2(cfg_path=os.path.join(model_dir, "config.yaml"),
                                   model_dir=model_dir, use_bf16=True)
        state["v25"] = True
    else:
        from indextts.infer_v2 import IndexTTS2
        state["model"] = IndexTTS2(cfg_path=os.path.join(model_dir, "config.yaml"), model_dir=model_dir)
        state["v25"] = False
    return {}


def synthesize(text, out, ref_audio=None, language=None, **_):
    if not ref_audio:
        raise ValueError("IndexTTS needs ref_audio")
    kwargs = {"spk_audio_prompt": ref_audio, "text": text, "output_path": None}
    if state["v25"] and language:
        kwargs["lang"] = language.upper()
    sr, wav = state["model"].infer(**kwargs)
    return write_wav(out, wav, sr)


serve({"load": load, "synthesize": synthesize})
