"""Upload and transcribe the bundled OpenDoor.wav example.

The temporary GCS object is deleted after transcription. Google Application
Default Credentials provide authentication.
"""

import argparse
from pathlib import Path

from ampav_gcp_pipeline import transcribe_file


INPUT_FILE = Path(__file__).parent / "data" / "OpenDoor.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("input_bucket", help="GCS bucket name or bucket-only gs:// URI")
    parser.add_argument("--location", default="us")
    parser.add_argument("--model", default="chirp_3")
    parser.add_argument("--language-code", action="append")
    parser.add_argument("--keep-input", action="store_true")
    args = parser.parse_args()

    output = transcribe_file(
        INPUT_FILE,
        project_id=args.project_id,
        input_bucket=args.input_bucket,
        location=args.location,
        model=args.model,
        language_codes=args.language_code or ("en-US",),
        keep_uploaded_input=args.keep_input,
    )
    print(output.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
