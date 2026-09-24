"""VoxCPM2 (OpenBMB) via the official ``voxcpm`` package."""
from _protocol import serve, write_wav

state = {}


def load(repo, params=None):
    import torch
    from voxcpm import VoxCPM

    # torch.compile ("optimize") is only exercised on CUDA upstream.
    state["model"] = VoxCPM.from_pretrained(repo, load_denoiser=False, optimize=torch.cuda.is_available())
    return {}


def synthesize(text, out, instruct=None, ref_audio=None, ref_text=None, **_):
    m = state["model"]
    if instruct:  # voice design / style: description in parentheses
        text = f"({instruct}){text}"
    kwargs = {}
    if ref_audio:
        kwargs["reference_wav_path"] = ref_audio
        if ref_text:
            kwargs.update(prompt_wav_path=ref_audio, prompt_text=ref_text)
    wav = m.generate(text=text, **kwargs)
    return write_wav(out, wav, m.tts_model.sample_rate)


serve({"load": load, "synthesize": synthesize})
