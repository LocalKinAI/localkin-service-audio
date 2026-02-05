"""
Benchmark command - Compare model performance.
"""
import click
import time
from typing import List, Optional

from ..utils import print_success, print_error, print_info, print_header


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--models", "-m",
    multiple=True,
    default=["whisper-cpp:tiny", "whisper-cpp:base", "faster-whisper:base"],
    help="Models to benchmark (can specify multiple)."
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=1,
    help="Number of iterations per model."
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file for results (JSON)."
)
def benchmark(
    audio_file: str,
    models: tuple,
    iterations: int,
    output: Optional[str]
):
    """
    Benchmark STT models for speed and accuracy.

    Examples:

        kin audio benchmark audio.wav

        kin audio benchmark audio.wav -m whisper-cpp:tiny -m whisper-cpp:base -m faster-whisper:small

        kin audio benchmark audio.wav --iterations 3 --output results.json
    """
    print_header("Model Benchmark")
    print_info(f"Audio file: {audio_file}")
    print_info(f"Models: {', '.join(models)}")
    print_info(f"Iterations: {iterations}")
    print()

    # Get audio duration
    audio_duration = _get_audio_duration(audio_file)
    print_info(f"Audio duration: {audio_duration:.2f}s")
    print()

    results = []

    from ...core import get_audio_engine
    engine = get_audio_engine()

    for model in models:
        print(f"\n📊 Benchmarking: {model}")
        print("-" * 40)

        model_results = {
            "model": model,
            "times": [],
            "rtf": [],
            "text": None,
        }

        try:
            # Load model
            load_start = time.time()
            engine.load_stt(model)
            load_time = time.time() - load_start
            print(f"  Load time: {load_time:.2f}s")

            # Run iterations
            for i in range(iterations):
                start = time.time()
                result = engine.transcribe(audio_file)
                elapsed = time.time() - start

                rtf = elapsed / audio_duration if audio_duration > 0 else 0
                model_results["times"].append(elapsed)
                model_results["rtf"].append(rtf)
                model_results["text"] = result.text

                print(f"  Run {i+1}: {elapsed:.2f}s (RTF: {rtf:.2f}x)")

            # Calculate averages
            avg_time = sum(model_results["times"]) / len(model_results["times"])
            avg_rtf = sum(model_results["rtf"]) / len(model_results["rtf"])

            model_results["avg_time"] = avg_time
            model_results["avg_rtf"] = avg_rtf
            model_results["load_time"] = load_time

            print(f"  Average: {avg_time:.2f}s (RTF: {avg_rtf:.2f}x)")

            results.append(model_results)

        except Exception as e:
            print_error(f"  Failed: {e}")
            model_results["error"] = str(e)
            results.append(model_results)

    # Summary
    print("\n" + "=" * 60)
    print("📈 Summary")
    print("=" * 60)
    print(f"\n{'MODEL':<30} {'AVG TIME':<12} {'RTF':<10} {'LOAD TIME'}")
    print("-" * 60)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<30} {'ERROR':<12}")
        else:
            print(f"{r['model']:<30} {r['avg_time']:.2f}s{'':<6} {r['avg_rtf']:.2f}x{'':<5} {r['load_time']:.2f}s")

    # Find fastest
    successful = [r for r in results if "error" not in r]
    if successful:
        fastest = min(successful, key=lambda x: x["avg_time"])
        print(f"\n🏆 Fastest: {fastest['model']} ({fastest['avg_time']:.2f}s)")

    # Save results
    if output:
        import json
        with open(output, "w") as f:
            json.dump({
                "audio_file": audio_file,
                "audio_duration": audio_duration,
                "iterations": iterations,
                "results": results,
            }, f, indent=2)
        print_success(f"Results saved to {output}")


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr
    except:
        try:
            import wave
            with wave.open(audio_path, 'r') as f:
                return f.getnframes() / float(f.getframerate())
        except:
            return 0.0
