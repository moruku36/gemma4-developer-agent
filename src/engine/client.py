"""Client for a local OpenAI-compatible chat endpoint (vLLM).

Implemented on the standard library only, so the scoring environment needs
no extra wheels: the offline bundle ships pyyaml and little else. The
:class:`ChatClient` protocol is what the agent loop depends on; tests
substitute a scripted fake implementing the same method.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient(Protocol):
    """Minimal interface the agent loop requires from any model backend."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        ...


class LocalVLLMClient:
    """HTTP client for vLLM's ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "gemma-4",
        timeout_seconds: float = 600.0,
        api_key: str = "EMPTY",  # vLLM ignores it, the API shape requires one
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(f"vLLM endpoint unreachable: {exc}") from exc
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return ChatResponse(
            text=choice["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
