"""Run and retain a native Google Natural Language classification probe."""

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

from ampav.gcp import GcpNaturalLanguageClassification


def _message_to_dict(message: Any) -> dict[str, Any]:
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
    classification = natural_language.get("classification", {})
    if not isinstance(classification, dict):
        raise ValueError("natural_language.classification must be a mapping")
    return config


def _response_observations(
    response: dict[str, Any], *, model: str, taxonomy: str
) -> str:
    categories = response.get("categories", [])
    category_lines = [
        f"- `{category.get('name', '')}`: confidence {category.get('confidence', 'not returned')}"
        for category in categories
    ]
    return "\n".join(
        [
            "# Observations",
            "",
            "## Native response structure",
            "",
            f"- Model: {model}",
            f"- Content taxonomy: {taxonomy}",
            f"- Categories: {len(categories)}",
            "- Each result preserves the provider category path and confidence.",
            "",
            "## Returned categories",
            "",
            *(category_lines or ["- None"]),
            "",
            "## Review needed",
            "",
            "- Compare specificity and confidence across the V1 and V2 runs.",
            "- Review category usefulness as controlled broad aboutness metadata.",
            "- Do not interpret category confidence as equivalent to entity salience or generative relevance.",
            "",
            "## Decision",
            "",
            "- Adopt the native classification wrapper for later AMPAV conversion work.",
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
    defaults = config.get("natural_language", {}).get("classification", {})
    language = args.language or config.get("natural_language", {}).get("language")
    model = args.model or defaults.get("model", "v2")
    taxonomy = args.content_categories_version or defaults.get(
        "content_categories_version", "v2"
    )
    effective_taxonomy = taxonomy if model == "v2" else "v1-only"

    started = datetime.now().astimezone()
    elapsed_started = monotonic()
    tool = GcpNaturalLanguageClassification()
    request = tool.build_request(
        text,
        language=language,
        model=model,
        content_categories_version=taxonomy,
    )
    response = tool.client.classify_text(request=request, timeout=args.timeout)
    native_request = _message_to_dict(request)
    native_response = _message_to_dict(response)

    (args.output_dir / "request.json").write_text(
        json.dumps(native_request, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "native_output.json").write_text(
        json.dumps(native_response, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "observations.md").write_text(
        _response_observations(
            native_response, model=model, taxonomy=effective_taxonomy
        ),
        encoding="utf-8",
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
        "model": model,
        "content_categories_version": effective_taxonomy,
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
    parser.add_argument("--model", choices=("v1", "v2"))
    parser.add_argument("--content-categories-version", choices=("v1", "v2"))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
