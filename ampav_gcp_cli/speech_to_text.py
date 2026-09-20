"""Command-line entry point for Google Cloud Speech-to-Text."""

import argparse
import logging
from os import PathLike
from pathlib import Path
from typing import Sequence

from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage

from ampav.core.async_tool import ToolError
from ampav.core.logging import LOG_FORMAT
from ampav.gcp import GcpSpeechToTextBatch
from ampav_gcp_pipeline.gcs_files import GcsLocation, upload_file


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the Speech-to-Text CLI parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe media with Google Cloud Speech-to-Text and print AMPAV ToolOutput YAML."
    )
    parser.add_argument("media", help="Local media path or gs://bucket/object media URI")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us", help="Speech-to-Text recognizer location")
    parser.add_argument("--model", default="chirp_3", help="Speech-to-Text recognition model")
    parser.add_argument(
        "--language-code",
        action="append",
        help="Recognition language code; repeat for multiple languages (default: en-US)",
    )
    parser.add_argument("--no-word-time-offsets", action="store_true", help="Disable word timestamps")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--input-bucket", help="GCS bucket for uploading a local media file")
    parser.add_argument("--input-object", help="Exact GCS object name for an uploaded local media file")
    parser.add_argument(
        "--input-prefix",
        default="gcp_speech_to_text/input",
        help="GCS prefix for generated input object names",
    )
    parser.add_argument("--keep-input", action="store_true", help="Keep the CLI-uploaded GCS object")
    parser.add_argument(
        "--include-tool-private",
        action="store_true",
        help="Include the native GCP response in ToolOutput.tool_private",
    )
    parser.add_argument("--poll-interval", type=float, default=30, help="Seconds between operation checks")
    parser.add_argument("--timeout", type=float, default=7200, help="Maximum seconds to wait for completion")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Google Cloud Speech-to-Text CLI."""
    args = build_cli_parser().parse_args(argv)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    speech = GcpSpeechToTextBatch(
        args.project_id,
        location=args.location,
        model=args.model,
        include_tool_private=args.include_tool_private,
        polling_interval=args.poll_interval,
        timeout=args.timeout,
    )
    storage_client = None
    uploaded_input = None
    try:
        media_uri = str(args.media)
        if not media_uri.startswith("gs://"):
            if not args.input_bucket:
                raise ValueError("--input-bucket is required when media is a local file")
            storage_client = storage.Client(project=args.project_id)
            media_uri, uploaded_input = _prepare_media_uri(
                storage_client,
                args.media,
                input_bucket=args.input_bucket,
                input_object=args.input_object,
                input_prefix=args.input_prefix,
            )

        result = speech.process(
            media_uri,
            language_codes=args.language_code or ("en-US",),
            enable_word_time_offsets=not args.no_word_time_offsets,
            enable_diarization=not args.no_diarization,
        )
    except Exception as exc:
        cli_errors = (ToolError, GoogleAPICallError, GoogleAuthError, OSError, ValueError)
        if not isinstance(exc, cli_errors):
            raise
        logging.error("%s", exc)
        return 1
    finally:
        if storage_client is not None and uploaded_input is not None and not args.keep_input:
            storage_client.bucket(uploaded_input.bucket).blob(uploaded_input.object_name).delete()

    print(result.model_dump_yaml(sort_keys=False))
    return 0


def _prepare_media_uri(
    storage_client: object,
    media: str | PathLike[str],
    *,
    input_bucket: str,
    input_object: str | None,
    input_prefix: str,
) -> tuple[str, GcsLocation]:
    location = upload_file(
        storage_client,
        Path(media),
        bucket=input_bucket,
        object_name=input_object,
        prefix=input_prefix,
        name_prefix="ampav-gcp-speech-to-text",
    )
    logging.info("Uploaded local media to %s", location.uri)
    return location.uri, location


if __name__ == "__main__":
    raise SystemExit(main())
