"""Character, word, exact-match, top-k, and subgroup OCR metrics."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .dataset import normalize_text
from .postprocessing import rank_candidates


def edit_distance(reference: str, hypothesis: str) -> int:
    """Compute Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def _aggregate(rows: list[dict[str, object]], *, lexicon: list[str] | None = None, top_ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, object]:
    references = [normalize_text(str(row["reference"])) for row in rows]
    hypotheses = [normalize_text(str(row["prediction"])) for row in rows]
    total_characters = sum(len(value) for value in references)
    total_words = sum(len(value.split()) for value in references)
    char_errors = sum(edit_distance(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses))
    word_errors = sum(
        edit_distance(reference.split(), hypothesis.split())
        for reference, hypothesis in zip(references, hypotheses)
    )
    result: dict[str, object] = {
        "samples": len(rows),
        "cer": char_errors / max(total_characters, 1),
        "wer": word_errors / max(total_words, 1),
        "exact_match_accuracy": sum(reference == hypothesis for reference, hypothesis in zip(references, hypotheses)) / max(len(rows), 1),
    }
    if lexicon:
        for top_k in top_ks:
            correct = 0
            for reference, hypothesis in zip(references, hypotheses):
                candidates = rank_candidates(hypothesis, lexicon, top_k=top_k)
                candidate_texts = {str(item["candidate"]) for item in candidates}
                correct += reference in candidate_texts
            result[f"top_{top_k}_accuracy"] = correct / max(len(rows), 1)
    return result


def evaluate_rows(
    rows: Iterable[dict[str, object]],
    *,
    training_label_counts: Counter[str] | None = None,
    rare_below: int = 5,
    lexicon: list[str] | None = None,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    """Compute overall and real/synthetic, frequent/rare subgroup metrics."""
    if rare_below < 1:
        raise ValueError("rare_below must be positive")
    materialized = list(rows)
    metrics: dict[str, object] = {"overall": _aggregate(materialized, lexicon=lexicon, top_ks=top_ks)}
    for group_key in ("sample_type",):
        groups: dict[str, list[dict[str, object]]] = {}
        for row in materialized:
            label = str(row.get(group_key) or "real")
            groups.setdefault(label, []).append(row)
        metrics["by_sample_type"] = {
            label: _aggregate(group_rows, lexicon=lexicon, top_ks=top_ks)
            for label, group_rows in sorted(groups.items())
        }

    if training_label_counts is not None:
        frequent, rare = [], []
        for row in materialized:
            reference = normalize_text(str(row["reference"]))
            destination = frequent if training_label_counts.get(reference, 0) >= rare_below else rare
            destination.append(row)
        metrics["frequency_threshold"] = {"rare_below_training_count": rare_below}
        metrics["by_frequency"] = {
            "frequent": _aggregate(frequent, lexicon=lexicon, top_ks=top_ks),
            "rare": _aggregate(rare, lexicon=lexicon, top_ks=top_ks),
        }
    return metrics

