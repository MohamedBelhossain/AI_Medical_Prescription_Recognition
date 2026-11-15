"""Prediction helpers for local TrOCR and the CRNN baseline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


@lru_cache(maxsize=2)
def _trocr_pipeline(model_name: str) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers and torch to use TrOCR inference.") from exc
    return pipeline("image-to-text", model=model_name, device=-1)


def predict_image(image: str | Path | Image.Image, model_name: str = "microsoft/trocr-base-handwritten") -> str:
    """Recognize a single handwritten word image with pretrained TrOCR."""
    try:
        from .preprocessing import load_image
    except ImportError:
        from preprocessing import load_image
    result = _trocr_pipeline(model_name)(load_image(image))
    return str(result[0]["generated_text"]).strip()


def greedy_ctc_decode(log_probs, alphabet: str) -> list[str]:
    """Decode (time, batch, classes) CTC log probabilities; class 0 is blank."""
    paths = log_probs.argmax(dim=2).detach().cpu().tolist()
    decoded: list[str] = []
    for path in zip(*paths):
        chars: list[str] = []
        previous = 0
        for token in path:
            if token and token != previous:
                if token - 1 >= len(alphabet):
                    raise ValueError("Model emitted a class outside the supplied alphabet")
                chars.append(alphabet[token - 1])
            previous = token
        decoded.append("".join(chars))
    return decoded
