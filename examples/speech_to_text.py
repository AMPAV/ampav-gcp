"""Transcribe caller-owned GCS audio into an AMPAV Transcript."""

import argparse

from ampav.gcp import GcpSpeechToTextBatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("gcs_uri")
    parser.add_argument("--location", default="us")
    args = parser.parse_args()

    tool = GcpSpeechToTextBatch(args.project_id, location=args.location)
    output = tool.process(args.gcs_uri)
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
