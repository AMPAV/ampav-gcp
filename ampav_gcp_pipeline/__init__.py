"""Blocking GCP pipeline adapters for connecting AMPAV tools."""

from .gcs_files import GcsLocation, upload_file
from .speech_to_text import transcribe_file

__all__ = ["GcsLocation", "transcribe_file", "upload_file"]
