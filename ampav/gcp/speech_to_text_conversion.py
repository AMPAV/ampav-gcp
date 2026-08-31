"""Convert native GCP Speech-to-Text output into AMPAV Transcript."""

from google.cloud.speech_v2.types import cloud_speech

from ampav.core.schema import ParagraphSegment, Transcript, WordSegment

from .errors import GcpSpeechToTextError, GcpTranscriptSchemaError


def gcp_speech_to_transcript(response: cloud_speech.BatchRecognizeResponse) -> Transcript:
    """Convert a one-file inline BatchRecognize response to Transcript."""
    if len(response.results) != 1:
        raise GcpTranscriptSchemaError("$.results", f"expected one file result, received {len(response.results)}")
    file_result = next(iter(response.results.values()))
    if file_result.error.code:
        raise GcpSpeechToTextError(file_result.uri or None, f"file failed: {file_result.error.message}")
    if not file_result._pb.HasField("inline_result"):
        raise GcpTranscriptSchemaError("$.results.*.inline_result", "inline transcript is required")

    words: list[WordSegment] = []
    paragraphs: list[ParagraphSegment] = []
    transcript_parts: list[str] = []
    languages: set[str] = set()
    for result in file_result.inline_result.transcript.results:
        if not result.alternatives:
            continue
        alternative = result.alternatives[0]
        result_words: list[WordSegment] = []
        for native_word in alternative.words:
            word = WordSegment.from_str(
                native_word.word,
                start_time=native_word.start_offset.total_seconds(),
                end_time=native_word.end_offset.total_seconds(),
                speaker=native_word.speaker_label or None,
                language=result.language_code or None,
                confidence=native_word.confidence or None,
            )
            result_words.append(word)
            words.append(word)
        if alternative.transcript.strip():
            transcript_parts.append(alternative.transcript.strip())
        if result.language_code:
            languages.add(result.language_code)
        if result_words:
            speakers = {word.speaker for word in result_words if word.speaker}
            paragraphs.append(
                ParagraphSegment(
                    start_time=result_words[0].start_time,
                    end_time=result_words[-1].end_time,
                    speaker=next(iter(speakers)) if len(speakers) == 1 else None,
                    language=result.language_code or None,
                    text=alternative.transcript.strip(),
                )
            )

    if not transcript_parts and not words:
        raise GcpTranscriptSchemaError("$.results.*.inline_result.transcript.results", "no recognition results were returned")
    media_duration = max((word.end_time for word in words if word.end_time is not None), default=None)
    return Transcript(
        text=" ".join(transcript_parts),
        media_duration=media_duration,
        words=words,
        paragraphs=paragraphs,
        languages=sorted(languages) or None,
    )
