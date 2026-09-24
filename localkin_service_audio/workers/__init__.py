"""
Worker scripts for models that need their own Python environment.

Each ``<family>.py`` runs under an isolated interpreter (see
core/audio_processing/isolated.py) and must import only the standard
library, ``_protocol`` and that family's own packages — never
localkin_service_audio, whose dependencies aren't installed there.
"""
