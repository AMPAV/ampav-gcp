"""Native synchronous Gemini generation through Vertex AI."""

from typing import Any

from google import genai
from google.genai import types


class GcpGeminiGenerateContent:
    """Generate content synchronously and return the native SDK response.

    The wrapper uses Vertex AI and Application Default Credentials. Prompt
    design, response schemas, persistence, and AMPAV schema conversion remain
    client responsibilities.
    """

    def __init__(
        self,
        project_id: str,
        *,
        location: str = "global",
        model: str = "gemini-2.5-flash",
        client: genai.Client | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("project_id must not be empty")
        if not location:
            raise ValueError("location must not be empty")
        if not model:
            raise ValueError("model must not be empty")

        self.project_id = project_id
        self.location = location
        self.model = model
        self.client = client or genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

    def build_config(
        self,
        *,
        system_instruction: str | None = None,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = 0.0,
        max_output_tokens: int | None = 4096,
    ) -> types.GenerateContentConfig:
        """Build native generation config, enabling JSON when given a schema."""
        if response_json_schema is not None and not response_json_schema:
            raise ValueError("response_json_schema must not be empty")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=(
                "application/json" if response_json_schema is not None else None
            ),
            response_json_schema=response_json_schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def process(
        self,
        contents: str,
        *,
        system_instruction: str | None = None,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = 0.0,
        max_output_tokens: int | None = 4096,
    ) -> types.GenerateContentResponse:
        """Generate content synchronously and return the native response."""
        if not contents.strip():
            raise ValueError("contents must not be empty")
        config = self.build_config(
            system_instruction=system_instruction,
            response_json_schema=response_json_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
