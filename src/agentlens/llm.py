"""Chat clients: an OpenAI-compatible HTTP client (stdlib only) and a deterministic mock model.

Any server that speaks ``POST /v1/chat/completions`` with ``tools`` works: OpenAI, Azure OpenAI,
vLLM, llama.cpp ``llama-server --jinja``, Ollama, LM Studio, OpenRouter, Together, Groq...
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from agentlens import _http

__all__ = [
    "LLMError",
    "ToolCallRequest",
    "ChatResponse",
    "ChatClient",
    "OpenAICompatClient",
    "MockChatClient",
    "MockRule",
    "estimate_tokens",
]


class LLMError(RuntimeError):
    """The model endpoint failed or returned something unusable."""


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: bool = False


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def assistant_message(self) -> dict[str, Any]:
        """The message to append to the conversation (OpenAI wire format)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        return msg


class ChatClient(Protocol):
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        model: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for clients that report no usage."""
    return max(1, (len(text) + 3) // 4) if text else 0


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str, bool]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw), False
    text = "" if raw is None else str(raw)
    if not text.strip():
        return {}, text, False
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"_raw": text}, text, True
    if not isinstance(parsed, dict):
        return {"_raw": text}, text, True
    return parsed, text, False


def parse_chat_response(data: dict[str, Any]) -> ChatResponse:
    """Parse an OpenAI chat-completions response body."""
    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"malformed chat completion: {str(data)[:300]}") from None
    calls = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args, raw, bad = _parse_arguments(fn.get("arguments"))
        calls.append(ToolCallRequest(str(tc.get("id") or f"call_{i}"), str(fn.get("name", "")), args, raw, bad))
    usage = data.get("usage") or {}
    return ChatResponse(
        content=message.get("content"),
        tool_calls=calls,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        model=str(data.get("model") or ""),
        finish_reason=str(choice.get("finish_reason") or ""),
        raw=data,
    )


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client (no SDK dependency), with retries on 429 / 5xx."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        model: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        body: dict[str, Any] = {"model": model, "messages": list(messages), "temperature": temperature}
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps(body).encode("utf-8")
        last_error = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", payload, headers, method="POST")
            try:
                with _http.urlopen(req, timeout=self.timeout_s) as resp:
                    return parse_chat_response(json.loads(resp.read().decode("utf-8")))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code != 429 and exc.code < 500:
                    break
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = str(exc)
            if attempt < self.max_retries:
                time.sleep(min(8.0, 1.0 * 2**attempt))
        raise LLMError(f"chat completion failed: {last_error}")


# ---- mock model ------------------------------------------------------------------------------
@dataclass
class MockRule:
    """When the first user message matches ``match`` (regex, case-insensitive), answer step by step.

    Each step is ``{"tool_calls": [{"name": ..., "arguments": {...}}]}`` or ``{"content": "..."}``.
    Step *n* is used when the conversation already holds *n* assistant messages.
    """

    match: str
    steps: list[dict[str, Any]]


class MockChatClient:
    """A deterministic, offline stand-in for an LLM, driven by regex rules (YAML or Python).

    Token usage is estimated from message lengths so cost accounting works end to end. This is what
    the quickstart uses so that everything runs with no API key and no model download.
    """

    def __init__(self, rules: Sequence[MockRule], default_content: str = "I don't know.", latency_s: float = 0.0):
        self.rules = list(rules)
        self.default_content = default_content
        self.latency_s = latency_s
        self.calls = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> MockChatClient:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rules = [MockRule(str(r["match"]), list(r["steps"])) for r in data.get("rules", [])]
        return cls(rules, default_content=str(data.get("default_content", "I don't know.")))

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        model: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        prompt_tokens = sum(estimate_tokens(json.dumps(m)) for m in messages) + (
            estimate_tokens(json.dumps(list(tools))) if tools else 0
        )
        step: dict[str, Any] = {"content": self.default_content}
        for rule in self.rules:
            if re.search(rule.match, user, re.IGNORECASE):
                step = rule.steps[n_assistant] if n_assistant < len(rule.steps) else {"content": self.default_content}
                break
        calls = [
            ToolCallRequest(f"call_{self.calls}_{i}", str(c["name"]), dict(c.get("arguments") or {}))
            for i, c in enumerate(step.get("tool_calls") or [])
        ]
        for c in calls:
            c.raw_arguments = json.dumps(c.arguments)
        content = step.get("content")
        out_text = (content or "") + "".join(c.raw_arguments + c.name for c in calls)
        return ChatResponse(
            content=content,
            tool_calls=calls,
            input_tokens=prompt_tokens,
            output_tokens=estimate_tokens(out_text),
            model=model,
            finish_reason="tool_calls" if calls else "stop",
        )
