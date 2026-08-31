"""Run and retain a native Speech-to-Text V2 batch recognition probe."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import sys
from time import monotonic
from typing import Any
from uuid import uuid4

from google.cloud import storage
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict
import yaml

from ampav.gcp import GcpSpeechToTextBatch


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert a proto-plus or protobuf message using provider field names."""
    pb_message = type(message).pb(message) if hasattr(type(message), "pb") else message
    return MessageToDict(pb_message, preserving_proto_field_name=True)


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("bucket must start with gs://")
    bucket_and_prefix = uri[5:]
    bucket, separator, prefix = bucket_and_prefix.partition("/")
    if not bucket:
        raise ValueError("bucket URI must include a bucket name")
    return bucket, prefix.rstrip("/") if separator else ""


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    if not isinstance(config.get("speech"), dict):
        raise ValueError("config must contain a speech mapping")
    for key in ("project", "bucket"):
        if not config.get(key):
            raise ValueError(f"config must contain {key}")
    for key in ("location", "model"):
        if not config["speech"].get(key):
            raise ValueError(f"speech config must contain {key}")
    return config


def _operation_details(operation: Any) -> dict[str, Any]:
    native_operation = getattr(operation, "operation", None)
    details: dict[str, Any] = {
        "name": getattr(native_operation, "name", None),
    }
    metadata = getattr(operation, "metadata", None)
    if metadata is not None:
        details["metadata"] = _message_to_dict(metadata)
    return details


def _file_errors(response: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (uri, file_result["error"])
        for uri, file_result in response.get("results", {}).items()
        if file_result.get("error")
    ]


def _response_observations(response: dict[str, Any]) -> str:
    file_results = response.get("results", {})
    speech_results: list[dict[str, Any]] = []
    file_errors = _file_errors(response)
    for file_result in file_results.values():
        inline_result = file_result.get("inline_result", {})
        transcript = inline_result.get("transcript", {})
        speech_results.extend(transcript.get("results", []))

    words: list[dict[str, Any]] = []
    languages: set[str] = set()
    for result in speech_results:
        if result.get("language_code"):
            languages.add(result["language_code"])
        alternatives = result.get("alternatives", [])
        if alternatives:
            words.extend(alternatives[0].get("words", []))

    speakers = sorted(
        {word["speaker_label"] for word in words if word.get("speaker_label")}
    )
    billed = response.get("total_billed_duration", "not returned")
    return "\n".join(
        [
            "# Observations",
            "",
            "## Native response structure",
            "",
            f"- File result entries: {len(file_results)}",
            f"- File errors: {len(file_errors)}",
            f"- Speech recognition results: {len(speech_results)}",
            f"- First-alternative words: {len(words)}",
            f"- Speaker labels: {', '.join(speakers) if speakers else 'none'}",
            f"- Result language codes: {', '.join(sorted(languages)) if languages else 'none'}",
            f"- Total billed duration: {billed}",
            *(
                [
                    "",
                    "## Native file errors",
                    "",
                    *[
                        f"- `{uri}`: code {error.get('code')} - "
                        f"{error.get('message', 'no message')}"
                        for uri, error in file_errors
                    ],
                ]
                if file_errors
                else []
            ),
            "",
            "## Review needed",
            "",
            "- Review transcript accuracy, punctuation, and capitalization.",
            "- Review word offsets and speaker-label usefulness.",
            "- Compare the native field structure with AMPAV transcript needs.",
            "- Record the adopt, defer, or reject decision after direct review.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    project = config["project"]
    bucket_uri = config["bucket"]
    location = config["speech"]["location"]
    model = config["speech"]["model"]
    bucket_name, prefix = _split_gcs_uri(bucket_uri)
    object_parts = [part for part in (prefix, "speech-to-text", uuid4().hex, input_path.name) if part]
    object_name = "/".join(object_parts)
    gcs_uri = f"gs://{bucket_name}/{object_name}"

    request_started = datetime.now().astimezone()
    elapsed_started = monotonic()
    storage_client = storage.Client(project=project)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    uploaded = False
    file_errors: list[tuple[str, dict[str, Any]]] = []

    tool = GcpSpeechToTextBatch(
        project,
        location=location,
        model=model,
    )
    request = tool.build_request(
        gcs_uri,
        language_codes=args.language_code,
        enable_word_time_offsets=not args.no_word_time_offsets,
        enable_diarization=not args.no_diarization,
    )
    (args.output_dir / "request.json").write_text(
        json.dumps(_message_to_dict(request), indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        blob.upload_from_filename(input_path)
        uploaded = True
        operation = tool.client.batch_recognize(request=request)
        response = operation.result(timeout=args.timeout)
        operation_details = _operation_details(operation)
        native_response = _message_to_dict(response)
        file_errors = _file_errors(native_response)
        (args.output_dir / "operation.json").write_text(
            json.dumps(operation_details, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "native_output.json").write_text(
            json.dumps(native_response, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "observations.md").write_text(
            _response_observations(native_response),
            encoding="utf-8",
        )
    finally:
        if uploaded and not args.keep_upload:
            blob.delete()

    manifest = {
        "timestamp": request_started.isoformat(),
        "elapsed_seconds": round(monotonic() - elapsed_started, 3),
        "fixture_id": input_path.stem,
        "google_cloud_speech_version": version("google-cloud-speech"),
        "google_cloud_storage_version": version("google-cloud-storage"),
        "project": project,
        "location": location,
        "model": model,
        "language_codes": args.language_code,
        "word_time_offsets": not args.no_word_time_offsets,
        "diarization": not args.no_diarization,
        "input_gcs_uri": gcs_uri,
        "temporary_upload_deleted": not args.keep_upload,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"local_input={input_path}\n"
        f"gcs_input={gcs_uri}\n"
        f"temporary_upload_deleted={not args.keep_upload}\n",
        encoding="utf-8",
    )
    if file_errors:
        error_summary = "; ".join(
            f"{uri}: code {error.get('code')} - "
            f"{error.get('message', 'no message')}"
            for uri, error in file_errors
        )
        raise RuntimeError(f"BatchRecognize returned file errors: {error_summary}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--language-code",
        action="append",
        default=None,
        help="BCP-47 language code; repeat for multiple values (default: en-US)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-word-time-offsets", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--keep-upload", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.language_code is None:
        args.language_code = ["en-US"]
    run(args)


if __name__ == "__main__":
    main()
