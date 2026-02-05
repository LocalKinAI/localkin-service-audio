"""
CLI utility modules.
"""
from .progress import OllamaStyleProgress
from .device import detect_hardware, recommend_models
from .output import (
    print_success,
    print_error,
    print_warning,
    print_info,
    print_header,
    print_model_table,
    format_duration,
    format_size,
)

__all__ = [
    "OllamaStyleProgress",
    "detect_hardware",
    "recommend_models",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_header",
    "print_model_table",
    "format_duration",
    "format_size",
]
