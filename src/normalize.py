"""Lightweight token-level normalization for social-media NER."""

from __future__ import annotations

import re
from typing import Iterable

USERNAME_RE = re.compile(r"^@[A-Za-z0-9_]+$")
URL_RE = re.compile(r"^(https?://\S+|www\.\S+)$", flags=re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"([A-Za-z])\1{2,}")


def collapse_repeated_chars(token: str, min_repeats: int = 3, keep: int = 2) -> str:
    """Collapse long repeated-character runs conservatively."""
    pattern = re.compile(rf"([A-Za-z])\1{{{max(min_repeats - 1, 1)},}}")
    return pattern.sub(lambda match: match.group(1) * keep, token)


def normalize_token(token: str, cfg: object) -> str:
    """Normalize a single token while preserving token boundaries."""
    normalized = token

    if getattr(cfg, "normalize_usernames", True) and USERNAME_RE.match(normalized):
        return "@USER"

    if getattr(cfg, "normalize_urls", True) and URL_RE.match(normalized):
        return "HTTPURL"

    if getattr(cfg, "collapse_repeated_chars", True):
        normalized = collapse_repeated_chars(
            normalized,
            min_repeats=int(getattr(cfg, "repeated_char_min_repeats", 3)),
            keep=int(getattr(cfg, "repeated_char_keep", 2)),
        )

    if getattr(cfg, "normalize_hashtags", False) and normalized.startswith("#") and len(normalized) > 1:
        normalized = normalized[1:]

    return normalized


def normalize_tokens(tokens: Iterable[str], cfg: object) -> list[str]:
    """Normalize a token list."""
    return [normalize_token(token, cfg) for token in tokens]


def is_social_noise_token(token: str) -> bool:
    """Heuristic flag for tokens likely affected by social-media noise."""
    return bool(
        USERNAME_RE.match(token)
        or URL_RE.match(token)
        or token.startswith("#")
        or REPEATED_CHAR_RE.search(token)
    )

