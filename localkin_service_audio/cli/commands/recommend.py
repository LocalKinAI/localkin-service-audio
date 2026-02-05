"""
Recommend command - Hardware-aware model recommendations.
"""
import click

from ..utils import print_header
from ..utils.device import detect_hardware, print_hardware_info, print_recommendations


@click.command()
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed hardware information."
)
def recommend(verbose: bool):
    """
    Recommend models based on your hardware.

    Detects your GPU, RAM, and CPU to suggest optimal models.

    Examples:

        kin audio recommend

        kin audio recommend --verbose
    """
    print_header("Model Recommendations")

    hardware = detect_hardware()

    if verbose:
        print_hardware_info(hardware)

    print_recommendations(hardware)
