"""Run and retain a structured Gemini aboutness-metadata probe."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import sys
from time import monotonic
from typing import Any

import yaml

from ampav.gcp import GcpGeminiGenerateContent


MODES = ("combined", "subjects", "topics", "themes", "categories")

SYSTEM_INSTRUCTION = (
    "Generate concise library discovery metadata that is faithful to the "
    "transcript. Describe what the item is actually about, not merely frequent "
    "words. Do not add facts that the transcript does not support."
)


def _array_schema(description: str, *, max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "maxItems": max_items,
        "items": {"type": "string"},
    }


ABOUTNESS_PROPERTIES: dict[str, dict[str, Any]] = {
    "summary": {
        "type": "string",
        "description": "A factual, concise summary of the audiovisual item.",
    },
    "subjects": _array_schema(
        "Catalog-like concepts, people, organizations, places, or events the item is primarily about.",
        max_items=8,
    ),
    "topics": _array_schema(
        "Specific matters or areas of discussion that receive substantive attention.",
        max_items=10,
    ),
    "themes": _array_schema(
        "Recurring abstract ideas, concerns, or interpretive threads in the item.",
        max_items=8,
    ),
    "categories": _array_schema(
        "Broad high-level knowledge or collection domains applicable to the item.",
        max_items=6,
    ),
    "keyphrases": _array_schema(
        "Concise salient phrases grounded in the transcript and useful for discovery.",
        max_items=15,
    ),
    "named_entities": {
        "type": "array",
        "description": "Important named entities explicitly present in the transcript.",
        "maxItems": 30,
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {
                    "type": "string",
                    "description": "The entity type in ordinary descriptive terms.",
                },
            },
            "required": ["name", "type"],
            "additionalProperties": False,
        },
    },
    "controlled_labels": {
        "type": "array",
        "description": "Only applicable labels from the caller-supplied vocabulary.",
        "maxItems": 6,
        "items": {
            "type": "string",
            "enum": [
                "arts and culture",
                "community",
                "education",
                "history",
                "science and technology",
                "sports",
            ],
        },
    },
}


def _response_schema(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    properties = (
        ABOUTNESS_PROPERTIES
        if mode == "combined"
        else {mode: ABOUTNESS_PROPERTIES[mode]}
    )
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    for key in ("project", "gemini"):
        if not config.get(key):
            raise ValueError(f"config must contain {key}")
    if not isinstance(config["gemini"], dict):
        raise ValueError("gemini config must be a mapping")
    for key in ("location", "model"):
        if not config["gemini"].get(key):
            raise ValueError(f"gemini config must contain {key}")
    return config


def _response_observations(native: dict[str, Any], *, mode: str) -> str:
    native_parsed = native.get("parsed")
    candidates = native.get("candidates", [])
    first_candidate = candidates[0] if candidates else {}
    parts = first_candidate.get("content", {}).get("parts", [])
    candidate_text = "".join(part.get("text", "") for part in parts)
    try:
        candidate_json = json.loads(candidate_text)
    except (json.JSONDecodeError, TypeError):
        candidate_json = None
    usage = native.get("usage_metadata", {})
    fields = sorted(candidate_json) if isinstance(candidate_json, dict) else []
    return "\n".join(
        [
            "# Observations",
            "",
            "## Native response structure",
            "",
            f"- Probe mode: {mode}",
            f"- Model version: {native.get('model_version', 'not returned')}",
            f"- Response ID returned: {bool(native.get('response_id'))}",
            f"- Finish reason: {first_candidate.get('finish_reason', 'not returned')}",
            f"- Candidate text is valid JSON: {candidate_json is not None}",
            f"- Candidate JSON fields: {', '.join(fields) if fields else 'none'}",
            f"- Native SDK parsed field returned: {native_parsed is not None}",
            f"- Prompt tokens: {usage.get('prompt_token_count', 'not returned')}",
            f"- Candidate tokens: {usage.get('candidates_token_count', 'not returned')}",
            f"- Thinking tokens: {usage.get('thoughts_token_count', 'not returned')}",
            f"- Total tokens: {usage.get('total_token_count', 'not returned')}",
            "- The native response retains candidates, finish reason, safety ratings, model version, response ID, and token usage.",
            "",
            "## Review needed",
            "",
            "- Review factual grounding, specificity, redundancy, and discovery value.",
            "- Compare focused terminology outputs with the combined response.",
            "- Compare generated entities and keyphrases with specialized tool output without treating their scores as equivalent.",
            "",
            "## Decision",
            "",
            "- Adopt synchronous native Gemini generation with caller-owned prompts and structured response schemas.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transcript = input_path.read_text(encoding="utf-8")
    gemini = config["gemini"]
    temperature = args.temperature if args.temperature is not None else gemini.get("temperature", 0.0)
    max_output_tokens = args.max_output_tokens or gemini.get("max_output_tokens", 4096)
    schema = _response_schema(args.mode)
    contents = f"Analyze this transcript for archival discovery metadata.\n\nTranscript:\n{transcript}"

    started = datetime.now().astimezone()
    elapsed_started = monotonic()
    tool = GcpGeminiGenerateContent(
        config["project"],
        location=gemini["location"],
        model=gemini["model"],
    )
    response = tool.process(
        contents,
        system_instruction=SYSTEM_INSTRUCTION,
        response_json_schema=schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    native_response = response.model_dump(mode="json", exclude_none=True)

    (args.output_dir / "response_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "native_output.json").write_text(
        json.dumps(native_response, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "observations.md").write_text(
        _response_observations(native_response, mode=args.mode), encoding="utf-8"
    )
    manifest = {
        "timestamp": started.isoformat(),
        "elapsed_seconds": round(monotonic() - elapsed_started, 3),
        "fixture_id": input_path.stem,
        "input_bytes": len(transcript.encode("utf-8")),
        "google_genai_version": version("google-genai"),
        "backend": "Vertex AI",
        "project": config["project"],
        "location": gemini["location"],
        "requested_model": gemini["model"],
        "returned_model_version": native_response.get("model_version"),
        "mode": args.mode,
        "response_mime_type": "application/json",
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"local_input={input_path}\nprovider_input=inline prompt text\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="combined")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
