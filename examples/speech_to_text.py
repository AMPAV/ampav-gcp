"""Transcribe caller-owned GCS audio into an AMPAV Transcript.

The example uses Google Application Default Credentials and never modifies the
input object.
"""

import argparse

from ampav.gcp import GcpSpeechToTextBatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("gcs_uri")
    parser.add_argument("--location", default="us")
    parser.add_argument("--model", default="chirp_3")
    parser.add_argument("--language-code", action="append")
    args = parser.parse_args()

    tool = GcpSpeechToTextBatch(args.project_id, location=args.location, model=args.model)
    output = tool.process(args.gcs_uri, language_codes=args.language_code or ("en-US",))
    print(output.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
