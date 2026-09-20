"""Client-side GCS helpers for GCP pipeline adapters, examples, and CLI."""

from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
import re


@dataclass(frozen=True)
class GcsLocation:
    """Location of one Google Cloud Storage object."""

    bucket: str
    object_name: str

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}"


def upload_file(
    storage_client: object,
    source: str | PathLike[str],
    *,
    bucket: str,
    object_name: str | None = None,
    prefix: str = "",
    name_prefix: str = "ampav-gcp",
) -> GcsLocation:
    """Upload a local file to GCS and return its object location."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    bucket_name = normalize_bucket_name(bucket)
    if object_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_prefix = _safe_object_part(name_prefix) or "ampav-gcp"
        safe_stem = _safe_object_part(source_path.stem) or "input"
        object_name = _join_object_name(
            prefix,
            f"{safe_prefix}-{timestamp}-{safe_stem}{source_path.suffix}",
        )
    elif not object_name.strip("/"):
        raise ValueError("object_name must not be empty")
    else:
        object_name = object_name.strip("/")

    storage_client.bucket(bucket_name).blob(object_name).upload_from_filename(str(source_path))
    return GcsLocation(bucket=bucket_name, object_name=object_name)


def normalize_bucket_name(value: str) -> str:
    """Accept a bucket name or bucket-only gs:// URI and return the name."""
    bucket = value.strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:].rstrip("/")
    if not bucket or "/" in bucket:
        raise ValueError("bucket must be a bucket name or bucket-only gs:// URI")
    return bucket


def _join_object_name(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _safe_object_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
