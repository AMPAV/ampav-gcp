"""GCP provider errors."""

from ampav.core.async_tool import ToolError


class GcpSpeechToTextError(ToolError):
    """Raised when the GCP Speech-to-Text workflow fails."""

    def __init__(self, job_id: str | None, message: str):
        self.job_id = job_id
        prefix = f"GCP Speech-to-Text job {job_id}: " if job_id else "GCP Speech-to-Text: "
        super().__init__(prefix + message)


class GcpTranscriptSchemaError(GcpSpeechToTextError):
    """Raised when native GCP output cannot be converted safely."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(None, f"{path}: {message}")
