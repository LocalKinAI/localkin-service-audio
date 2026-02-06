"""
Add-model command - Add a new model to LocalKin Audio.
"""
import click

from ..utils import print_header, print_success, print_error, print_info, print_warning


@click.command("add-model")
@click.option("--template", "template_name", default=None, help="Use a model template (see list-templates).")
@click.option("--repo", default=None, help="Hugging Face repository (org/model).")
@click.option("--name", "model_name", required=True, help="Name for the new model.")
@click.option("--description", default=None, help="Description of the model.")
@click.option("--type", "model_type", type=click.Choice(["stt", "tts"]), default=None, help="Model type.")
@click.option("--license", "model_license", default="MIT", help="Model license.")
@click.option("--size-mb", type=int, default=500, help="Approximate model size in MB.")
@click.option("--tags", default=None, help="Comma-separated tags for the model.")
def add_model(template_name, repo, model_name, description, model_type, model_license, size_mb, tags):
    """
    Add a new model to LocalKin Audio.

    Provide either --template or --repo to specify the model source.

    Examples:

        kin audio add-model --template whisper_stt --name my-whisper

        kin audio add-model --repo openai/whisper-medium --name whisper-med --type stt

        kin audio add-model --repo microsoft/speecht5_tts --name my-tts --type tts --tags neural,microsoft
    """
    if not template_name and not repo:
        print_error("Either --template or --repo is required.")
        print_info("See available templates: kin audio list-templates")
        raise click.Abort()

    if template_name and repo:
        print_error("Cannot use both --template and --repo. Choose one.")
        raise click.Abort()

    print_header("Add Model")

    from ...core.config_legacy import get_models, get_config_metadata, save_models_config

    current_models = get_models()
    current_names = [m.get("name") for m in current_models]

    # Check for duplicate
    if model_name in current_names:
        if not click.confirm(f"Model '{model_name}' already exists. Overwrite?"):
            print_info("Cancelled.")
            return
        current_models = [m for m in current_models if m.get("name") != model_name]

    model = None

    if template_name:
        from ...templates.model_templates import list_available_templates, create_model_from_template

        available = list_available_templates()
        if template_name not in available:
            print_error(f"Template '{template_name}' not found.")
            print_info("Available templates:")
            for t in available:
                print(f"  - {t}")
            raise click.Abort()

        try:
            model = create_model_from_template(template_name, model_name, description, repo)
        except Exception as e:
            print_error(f"Failed to create model from template: {e}")
            raise click.Abort()

    elif repo:
        # Auto-detect type from repo name
        detected_type = "stt"
        repo_lower = repo.lower()
        if any(kw in repo_lower for kw in ["tts", "speech", "bark", "tacotron", "fastspeech"]):
            detected_type = "tts"
        if any(kw in repo_lower for kw in ["whisper", "wav2vec", "hubert", "stt", "asr"]):
            detected_type = "stt"

        final_type = model_type or detected_type

        model = {
            "name": model_name,
            "type": final_type,
            "description": description or f"Custom {final_type.upper()} model from Hugging Face",
            "source": "huggingface",
            "huggingface_repo": repo,
            "license": model_license,
            "size_mb": size_mb,
            "requirements": ["transformers", "torch"],
            "tags": tags.split(",") if tags else ["custom", "huggingface"],
        }

    if not model:
        print_error("Could not create model configuration.")
        raise click.Abort()

    # Validate if HuggingFace model
    if model.get("source") == "huggingface":
        from ...templates.model_templates import validate_model_for_huggingface
        warnings = validate_model_for_huggingface(model)
        if warnings:
            print_warning("Validation warnings:")
            for w in warnings:
                print(f"  - {w}")

    # Save
    current_models.append(model)
    metadata = get_config_metadata()
    if save_models_config(current_models, metadata):
        print_success(f"Model '{model_name}' added successfully!")
        print(f"\n  Type:   {model.get('type', 'N/A').upper()}")
        print(f"  Source: {model.get('source', 'N/A')}")
        if model.get("huggingface_repo"):
            print(f"  Repo:   {model['huggingface_repo']}")
        print(f"  Size:   {model.get('size_mb', 'N/A')} MB")
        print(f"\n  Next steps:")
        print(f"    kin audio models")
        print(f"    kin audio serve {model_name} --port 8000")
    else:
        print_error("Failed to save model configuration.")
