"""HTTP helpers shared by the OpenAI-compatible client and the OTLP exporter."""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_loopback(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in _LOOPBACK or host.startswith("127.")


def urlopen(req: urllib.request.Request, timeout: float) -> Any:
    """``urllib.request.urlopen`` that never sends loopback traffic through an HTTP(S) proxy taken
    from the environment (a local llama.cpp server or OTLP collector must be reached directly)."""
    if is_loopback(req.full_url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)
