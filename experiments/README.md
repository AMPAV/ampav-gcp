# GCP experiments

## Natural Language entity analysis

`natural_language_entities.py` sends one direct text input to the Natural
Language V1 entity-analysis API and retains the native request and response.

```bash
python experiments/natural_language_entities.py \
  --config CONFIG.yaml \
  --input TRANSCRIPT.txt \
  --output-dir RUN_DIRECTORY
```

The config supplies `project` for the retained manifest and may supply a
`natural_language.language` default. Authentication uses Application Default
Credentials and its configured quota project.

## Speech-to-Text

`speech_to_text.py` uploads one local audio file to a caller-selected GCS
bucket, runs Speech-to-Text V2 batch recognition, retains the native request,
operation metadata, and response, and then deletes the temporary upload.
Per-file provider errors are retained in the same run record and cause the
probe to exit nonzero.

Install the experiment dependencies:

```bash
python -m pip install -e '.[experiments]'
```

Run the probe with local config and output paths:

```bash
python experiments/speech_to_text.py \
  --config CONFIG.yaml \
  --input AUDIO.m4a \
  --output-dir RUN_DIRECTORY
```

The config supplies `project`, `bucket`, and the `speech.location` and
`speech.model` defaults. Authentication uses Application Default Credentials;
credentials and local configuration do not belong in this repository.
