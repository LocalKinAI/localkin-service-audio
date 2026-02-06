"""
List-templates command - List available model templates.
"""
import click

from ..utils import print_header, print_success


@click.command("list-templates")
def list_templates():
    """
    List available model templates.

    Shows all templates that can be used with the add-model command,
    plus popular pre-configured models.

    Examples:

        kin audio list-templates
    """
    print_header("Model Templates")

    from ...templates.model_templates import (
        list_available_templates,
        get_model_template,
        get_popular_models,
    )

    templates = list_available_templates()

    print(f"\n📚 All Templates ({len(templates)}):")
    print("-" * 50)

    for name in templates:
        template = get_model_template(name)
        if template:
            print(f"\n  📦 {name}")
            print(f"     Type:        {template['type'].upper()}")
            print(f"     Size:        {template.get('size_mb', 'N/A')} MB")
            print(f"     Description: {template.get('description', 'N/A')}")
            if "tags" in template:
                print(f"     Tags:        {', '.join(template['tags'])}")

    popular = get_popular_models()
    if popular:
        print(f"\n\n🚀 Popular Models (Ready to use):")
        print("-" * 40)
        for model in popular:
            print(f"\n  ⭐ {model['name']}")
            print(f"     {model.get('description', 'N/A')}")

    print(f"\n💡 Usage:")
    print(f"  kin audio add-model --template whisper_stt --name my-whisper")
    print(f"  kin audio add-model --repo openai/whisper-medium --name whisper-med --type stt")
