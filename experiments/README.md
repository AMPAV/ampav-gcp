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

## Natural Language content classification

`natural_language_classification.py` sends one direct text input to the
Natural Language classify-text API and retains one native model response. Run
it separately for V1 and V2 to preserve comparable provider outputs:

```bash
python experiments/natural_language_classification.py \
  --config CONFIG.yaml \
  --input TRANSCRIPT.txt \
  --model v2 \
  --content-categories-version v2 \
  --output-dir RUN_DIRECTORY
```

The config may supply defaults under `natural_language.classification`.

## Gemini aboutness metadata

`gemini_aboutness.py` sends transcript text synchronously to Gemini on Vertex
AI, applies a caller-owned JSON schema, and retains the complete native SDK
response. The `combined` mode requests all candidate metadata. Focused modes
allow direct comparison of `subjects`, `topics`, `themes`, and `categories`.

```bash
python experiments/gemini_aboutness.py \
  --config CONFIG.yaml \
  --input TRANSCRIPT.txt \
  --mode combined \
  --output-dir RUN_DIRECTORY
```

The config supplies `gemini.location`, `gemini.model`, and optional generation
defaults. Authentication uses Application Default Credentials.

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
