# ampav-gcp

Google Cloud tooling for the AMPAV environment.

## Speech-to-Text

`GcpSpeechToTextBatch` wraps the native Speech-to-Text V2 asynchronous batch
lifecycle and converts completed Chirp 3 output to
`ToolOutput.output = Transcript`:

```python
from ampav.gcp import GcpSpeechToTextBatch

tool = GcpSpeechToTextBatch("project-id", location="us")
output = tool.process("gs://bucket/audio.wav")
print(output.output.text)
```

The explicit lifecycle is `submit`, `get_status`, `get_result`, `cleanup`, and
`list_jobs`. Inputs must already exist in GCS and remain caller-owned. Google
Application Default Credentials provide authentication. Set
`include_tool_private=True` only when the native provider response is needed
for troubleshooting.

The blocking pipeline adapter owns the upload and default cleanup needed for a
local media file:

```python
from ampav_gcp_pipeline import transcribe_file

output = transcribe_file(
    "audio.wav",
    project_id="project-id",
    input_bucket="bucket-name",
)
```

The CLI accepts either a caller-owned GCS URI or a local path. Local paths
require a temporary input bucket:

```bash
ampav_gcp_speech_to_text gs://bucket/audio.wav --project-id project-id
ampav_gcp_speech_to_text audio.wav --project-id project-id --input-bucket bucket-name
```

See `examples/speech_to_text.py` for an existing GCS object and
`examples/speech_to_text_file.py` for a local-file upload using the bundled
OpenDoor fixture.

## Development

Use the shared AMPAV virtual environment and install the package in editable
mode:

```bash
python -m pip install -e .
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```
