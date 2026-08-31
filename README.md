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
