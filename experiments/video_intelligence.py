"""Run and retain a combined native Video Intelligence annotation probe."""

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
from google.protobuf.json_format import MessageToDict
import yaml

from ampav.gcp import GcpVideoIntelligence
from ampav.gcp.video_intelligence import DEFAULT_FEATURES


FEATURES_BY_NAME = {feature.name: feature for feature in DEFAULT_FEATURES}


def _message_to_dict(message: Any) -> dict[str, Any]:
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
    for key in ("project", "bucket", "video_intelligence"):
        if not config.get(key):
            raise ValueError(f"config must contain {key}")
    video = config["video_intelligence"]
    if not isinstance(video, dict):
        raise ValueError("video_intelligence config must be a mapping")
    for key in ("location", "model"):
        if not video.get(key):
            raise ValueError(f"video_intelligence config must contain {key}")
    return config


def _operation_details(operation: Any) -> dict[str, Any]:
    native_operation = getattr(operation, "operation", None)
    details: dict[str, Any] = {"name": getattr(native_operation, "name", None)}
    metadata = getattr(operation, "metadata", None)
    if metadata is not None:
        details["metadata"] = _message_to_dict(metadata)
    return details


def _response_observations(response: dict[str, Any]) -> str:
    results = response.get("annotation_results", [])
    errors = [result["error"] for result in results if result.get("error")]
    faces = [
        annotation
        for result in results
        for annotation in result.get("face_detection_annotations", [])
    ]
    people = [
        annotation
        for result in results
        for annotation in result.get("person_detection_annotations", [])
    ]
    texts = [
        annotation
        for result in results
        for annotation in result.get("text_annotations", [])
    ]
    objects = [
        annotation
        for result in results
        for annotation in result.get("object_annotations", [])
    ]
    face_tracks = [track for annotation in faces for track in annotation.get("tracks", [])]
    person_tracks = [track for annotation in people for track in annotation.get("tracks", [])]
    text_segments = [segment for annotation in texts for segment in annotation.get("segments", [])]
    decision = (
        "- Adopt the native asynchronous wrapper and successful feature outputs; defer the failing feature configuration."
        if errors
        else "- Adopt the native asynchronous wrapper and selected feature outputs for later AMPAV conversion work."
    )
    return "\n".join(
        [
            "# Observations",
            "",
            "## Combined native response structure",
            "",
            f"- Annotation result entries: {len(results)}",
            f"- Result errors: {errors or 'none'}",
            f"- Shot segments: {sum(len(result.get('shot_annotations', [])) for result in results)}",
            f"- Segment labels: {sum(len(result.get('segment_label_annotations', [])) for result in results)}",
            f"- Shot labels: {sum(len(result.get('shot_label_annotations', [])) for result in results)}",
            f"- Frame labels: {sum(len(result.get('frame_label_annotations', [])) for result in results)}",
            f"- OCR text annotations: {len(texts)}",
            f"- OCR text segments: {len(text_segments)}",
            f"- Face detection annotations: {len(faces)}",
            f"- Face tracks: {len(face_tracks)}",
            f"- Person detection annotations: {len(people)}",
            f"- Person tracks: {len(person_tracks)}",
            f"- Object tracks: {len(objects)}",
            "- Provider results group feature-specific arrays for labels, shots, OCR text, faces, people, and objects; a request can return populated and error result entries together.",
            "- Track outputs retain native segments, timestamped bounding boxes, confidence, attributes, and pose landmarks where returned.",
            "",
            "## Review needed",
            "",
            "- Review shot boundaries for navigation and segmentation value.",
            "- Review label specificity and OCR text accuracy.",
            "- Review face, person, and object tracks for useful temporal and spatial detail without inferring identity.",
            "- Confirm whether the combined lifecycle and grouped result favor one low-level video-analysis tool.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    video = config["video_intelligence"]
    selected_features = (
        [FEATURES_BY_NAME[name] for name in args.feature]
        if args.feature
        else list(DEFAULT_FEATURES)
    )
    include_bounding_boxes = not args.no_bounding_boxes
    include_attributes = not args.no_attributes
    include_pose_landmarks = args.pose_landmarks or (
        video.get("include_pose_landmarks", False) and not args.no_pose_landmarks
    )
    bucket_name, prefix = _split_gcs_uri(config["bucket"])
    object_parts = [
        part
        for part in (
            prefix,
            "video-intelligence",
            uuid4().hex,
            input_path.name,
        )
        if part
    ]
    object_name = "/".join(object_parts)
    gcs_uri = f"gs://{bucket_name}/{object_name}"

    started = datetime.now().astimezone()
    elapsed_started = monotonic()
    storage_client = storage.Client(project=config["project"])
    blob = storage_client.bucket(bucket_name).blob(object_name)
    uploaded = False

    tool = GcpVideoIntelligence(
        location=video["location"],
        model=video["model"],
    )
    request = tool.build_request(
        gcs_uri,
        features=selected_features,
        include_bounding_boxes=include_bounding_boxes,
        include_attributes=include_attributes,
        include_pose_landmarks=include_pose_landmarks,
    )
    (args.output_dir / "request.json").write_text(
        json.dumps(_message_to_dict(request), indent=2) + "\n", encoding="utf-8"
    )

    try:
        blob.upload_from_filename(input_path)
        uploaded = True
        operation = tool.client.annotate_video(request=request)
        response = operation.result(timeout=args.timeout)
        (args.output_dir / "operation.json").write_text(
            json.dumps(_operation_details(operation), indent=2) + "\n",
            encoding="utf-8",
        )
        native_response = _message_to_dict(response)
        (args.output_dir / "native_output.json").write_text(
            json.dumps(native_response, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "observations.md").write_text(
            _response_observations(native_response), encoding="utf-8"
        )
    finally:
        if uploaded and not args.keep_upload:
            blob.delete()

    manifest = {
        "timestamp": started.isoformat(),
        "elapsed_seconds": round(monotonic() - elapsed_started, 3),
        "fixture_id": input_path.stem,
        "input_bytes": input_path.stat().st_size,
        "google_cloud_videointelligence_version": version(
            "google-cloud-videointelligence"
        ),
        "google_cloud_storage_version": version("google-cloud-storage"),
        "project": config["project"],
        "location": video["location"],
        "model": video["model"],
        "features": [feature.name for feature in selected_features],
        "include_bounding_boxes": include_bounding_boxes,
        "include_attributes": include_attributes,
        "include_pose_landmarks": include_pose_landmarks,
        "input_gcs_uri": gcs_uri,
        "temporary_upload_deleted": not args.keep_upload,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"local_input={input_path}\n"
        f"gcs_input={gcs_uri}\n"
        f"temporary_upload_deleted={not args.keep_upload}\n",
        encoding="utf-8",
    )
    errors = [
        result["error"]
        for result in native_response.get("annotation_results", [])
        if result.get("error")
    ]
    if errors:
        raise RuntimeError(f"AnnotateVideo returned result errors: {errors}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--feature",
        action="append",
        choices=tuple(FEATURES_BY_NAME),
        help="Feature to request; repeat for a subset (default: all selected features)",
    )
    parser.add_argument("--no-bounding-boxes", action="store_true")
    parser.add_argument("--no-attributes", action="store_true")
    parser.add_argument("--pose-landmarks", action="store_true")
    parser.add_argument("--no-pose-landmarks", action="store_true")
    parser.add_argument("--keep-upload", action="store_true")
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
