"""Vocabulary-based OCR candidate ranking.

Scores are normalized edit-similarity values, not calibrated probabilities or
medical confidence estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterable

from .dataset import normalize_text


def load_lexicon(path: str | Path | BinaryIO) -> list[str]:
    """Read one candidate per line, skipping blanks and exact duplicates."""
    values: list[str] = []
    seen: set[str] = set()
    if hasattr(path, "getvalue"):
        content = path.getvalue()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        lines = str(content).splitlines()
    else:
        with Path(path).open(encoding="utf-8-sig") as handle:
            lines = list(handle)
    for line in lines:
        candidate = normalize_text(line)
        if candidate and candidate not in seen:
            values.append(candidate)
            seen.add(candidate)
    return values


def rank_candidates(raw_text: str, lexicon: Iterable[str], *, top_k: int = 5) -> list[dict[str, float | str]]:
    """Return lexicon candidates sorted by normalized edit similarity."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query = normalize_text(raw_text)
    if not query:
        return []
    candidates = list(dict.fromkeys(normalize_text(item) for item in lexicon if normalize_text(item)))
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein

        matches = process.extract(query, candidates, scorer=Levenshtein.normalized_similarity, limit=top_k)
        ranked = [(candidate, float(score)) for candidate, score, _ in matches]
    except ImportError:
        ranked = [
            (candidate, _normalized_edit_similarity(query, candidate))
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        ranked = ranked[:top_k]
    return [
        {"candidate": candidate, "score": round(score, 6), "score_type": "normalized_edit_similarity"}
        for candidate, score in ranked
    ]


def _normalized_edit_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))

