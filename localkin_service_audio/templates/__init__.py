"""
Model templates for LocalKin Service Audio.

This module contains predefined templates for easily adding new models.
"""

from .model_templates import (
    get_model_template, list_available_templates, create_model_from_template,
    validate_model_for_huggingface, suggest_similar_models, get_popular_models
)

__all__ = [
    "get_model_template", "list_available_templates", "create_model_from_template",
    "validate_model_for_huggingface", "suggest_similar_models", "get_popular_models"
]
