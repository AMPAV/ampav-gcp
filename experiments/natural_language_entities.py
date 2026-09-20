"""Run and retain a native Google Natural Language entity-analysis probe."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import sys
from time import monotonic
from typing import Any

from google.protobuf.json_format import MessageToDict
import yaml

from ampav.gcp import GcpNaturalLanguageEntities


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert a proto-plus message using provider field names."""
    pb_message = type(message).pb(message)
    return MessageToDict(pb_message, preserving_proto_field_name=True)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    if not config.get("project"):
        raise ValueError("config must contain project")
    natural_language = config.get("natural_language", {})
    if not isinstance(natural_language, dict):
        raise ValueError("natural_language config must be a mapping")
    return config


def _response_observations(response: dict[str, Any]) -> str:
    entities = response.get("entities", [])
    type_counts: dict[str, int] = {}
    mention_count = 0
    metadata_count = 0
    for entity in entities:
        entity_type = entity.get("type_", "UNKNOWN")
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        mention_count += len(entity.get("mentions", []))
        metadata_count += bool(entity.get("metadata"))
    type_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(type_counts.items())
    )
    return "\n".join(
        [
            "# Observations",
            "",
            "## Native response structure",
            "",
            f"- Detected language: {response.get('language', 'not returned')}",
            f"- Entities: {len(entities)}",
            f"- Mentions: {mention_count}",
            f"- Entities with provider metadata: {metadata_count}",
            f"- Entity types: {type_summary or 'none'}",
            "- Each entity preserves native Python fields for name, type_, salience, metadata, and mentions.",
            "- Each mention preserves native Python fields for content, UTF-8 begin_offset, and type_.",
            "",
            "## Review needed",
            "",
            "- Review entity precision, omissions, type choices, and duplicate naming.",
            "- Compare salience semantics with AWS confidence and GLiNER scores without treating them as equivalent.",
            "- Compare the native structure with future AMPAV named-entity needs.",
            "",
            "## Decision",
            "",
            "- Adopt the native entity-analysis wrapper for later AMPAV conversion work.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    text = input_path.read_text(encoding="utf-8")
    language = args.language or config.get("natural_language", {}).get("language")

    started = datetime.now().astimezone()
    elapsed_started = monotonic()
    tool = GcpNaturalLanguageEntities()
    request = tool.build_request(text, language=language)
    response = tool.client.analyze_entities(request=request, timeout=args.timeout)
    native_request = _message_to_dict(request)
    native_response = _message_to_dict(response)

    (args.output_dir / "request.json").write_text(
        json.dumps(native_request, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "native_output.json").write_text(
        json.dumps(native_response, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "observations.md").write_text(
        _response_observations(native_response), encoding="utf-8"
    )
    manifest = {
        "timestamp": started.isoformat(),
        "elapsed_seconds": round(monotonic() - elapsed_started, 3),
        "fixture_id": input_path.stem,
        "input_bytes": len(text.encode("utf-8")),
        "google_cloud_language_version": version("google-cloud-language"),
        "api_version": "v1",
        "project": config["project"],
        "language": language or "auto-detect",
        "encoding_type": "UTF8",
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"local_input={input_path}\nprovider_input=inline text\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
