"""
Cache command - Manage model cache.
"""
import click

from ..utils import print_header, print_success, print_error, print_info, print_warning


@click.group(invoke_without_command=True)
@click.pass_context
def cache(ctx):
    """
    Manage model cache.

    View cache information or clear cached models.

    Examples:

        kin audio cache

        kin audio cache info

        kin audio cache clear

        kin audio cache clear whisper-large
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(cache_info)


@cache.command("info")
def cache_info():
    """
    Show cache information.

    Displays cache directory, cached models, and their sizes.
    """
    print_header("Cache Information")

    try:
        from ...core.models import get_cache_info
        info = get_cache_info()
    except Exception as e:
        print_error(f"Could not read cache info: {e}")
        raise click.Abort()

    print(f"\n📁 Cache directory: {info.get('cache_dir', 'N/A')}")
    print(f"📁 HuggingFace cache: {info.get('huggingface_cache', 'N/A')}")

    cached_models = info.get("cached_models", [])
    if cached_models:
        print(f"\n📦 Cached Models ({len(cached_models)}):")
        print("-" * 50)
        for model in cached_models:
            name = model.get("name", "Unknown")
            size = model.get("size_mb", 0)
            path = model.get("path", "")
            print(f"  {name:<30} {size:>8.1f} MB")
            if path:
                print(f"    {path}")
    else:
        print_info("No cached models found.")


@cache.command("clear")
@click.argument("model_name", required=False)
def cache_clear(model_name):
    """
    Clear cached models.

    If MODEL_NAME is provided, only that model's cache is cleared.
    Otherwise, prompts to clear all cached models.

    Examples:

        kin audio cache clear

        kin audio cache clear whisper-large
    """
    print_header("Clear Cache")

    from ...core.models import clear_cache

    if model_name:
        print_info(f"Clearing cache for: {model_name}")
        result = clear_cache(model_name)
    else:
        click.confirm(
            "⚠️  This will clear ALL cached models. Continue?",
            abort=True,
        )
        result = clear_cache()

    if result:
        print_success("Cache cleared successfully.")
    else:
        print_error("Failed to clear cache.")
