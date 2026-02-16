"""Base agent with shared LLM calling logic."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

from alethia.config import Config


class BaseAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = config.create_openai_client()

    def _call_llm(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.7,
        images: list[str] | None = None,
    ) -> str:
        model = model or self.config.generator_model

        # Build user message content — text-only or multimodal with images
        if images:
            user_content = _build_vision_content(user, images)
        else:
            user_content = user

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def _build_vision_content(text: str, image_paths: list[str]) -> list[dict]:
    """Build a multimodal content array with text + base64-encoded images."""
    parts: list[dict] = [{"type": "text", "text": text}]
    for img_path in image_paths:
        p = Path(img_path)
        if not p.is_file():
            continue
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        })
    return parts


def extract_code_block(text: str) -> str:
    """Extract code from markdown ```python fences, falling back to the full text."""
    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fences — return as-is (stripped)
    return text.strip()
