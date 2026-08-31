"""Blocking pipeline adapters for Google Cloud Speech-to-Text."""

from collections.abc import Sequence
from os import PathLike
from typing import Any

from google.cloud import storage

from ampav.core.schema import ToolOutput
from ampav.gcp import GcpSpeechToTextBatch

from .gcs_files import upload_file


_INPUT_PREFIX = "gcp_speech_to_text/input"


def transcribe_file(
    source: str | PathLike[str],
    *,
    project_id: str,
    input_bucket: str,
    input_prefix: str = _INPUT_PREFIX,
    input_object_name: str | None = None,
    location: str = "us",
    model: str = "chirp_3",
    language_codes: Sequence[str] = ("en-US",),
    enable_word_time_offsets: bool = True,
    enable_diarization: bool = True,
    include_tool_private: bool = False,
    polling_interval: float = 30,
    timeout: float | None = 7200,
    speech_client: Any | None = None,
    storage_client: Any | None = None,
    keep_uploaded_input: bool = False,
) -> ToolOutput:
    """Upload local media, transcribe it, and delete the uploaded object by default."""
    gcs = storage_client if storage_client is not None else storage.Client(project=project_id)
    speech = GcpSpeechToTextBatch(
        project_id,
        location=location,
        model=model,
        client=speech_client,
        include_tool_private=include_tool_private,
        polling_interval=polling_interval,
        timeout=timeout,
    )
    input_location = upload_file(
        gcs,
        source,
        bucket=input_bucket,
        object_name=input_object_name,
        prefix=input_prefix,
        name_prefix="ampav-gcp-speech-to-text",
    )
    try:
        return speech.process(
            input_location.uri,
            language_codes=language_codes,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_diarization=enable_diarization,
        )
    finally:
        if not keep_uploaded_input:
            gcs.bucket(input_location.bucket).blob(input_location.object_name).delete()
