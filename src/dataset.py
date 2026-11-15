"""Dataset discovery, manifest loading, normalization, and leakage-aware splits."""

from __future__ import annotations

import csv
import hashlib
import random
import unicodedata
from pathlib import Path
from typing import NamedTuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class Sample(NamedTuple):
    image_path: Path
    text: str
    source: str = ""
    split: str = ""
    writer_id: str = ""
    document_id: str = ""
    group_id: str = ""
    sample_type: str = "real"
    raw_text: str = ""
    generation_kind: str = ""
    font_name: str = ""
    font_license: str = ""


_PATH_COLUMNS = {
    "image",
    "image_path",
    "image_name",
    "image_file",
    "image_filename",
    "path",
    "filename",
    "file_name",
    "file",
    "img",
}
_TEXT_COLUMNS = {
    "text",
    "label",
    "transcription",
    "word",
    "word_text",
    "medicine",
    "medicine_name",
    "medication",
    "drug",
    "drug_name",
}


def _column_index(header: list[str], choices: set[str]) -> int | None:
    return next((index for index, value in enumerate(header) if value in choices), None)


def _cell(row: list[str], index: int | None) -> str:
    return row[index].strip() if index is not None and index < len(row) else ""


def load_manifest(path: str | Path) -> list[Sample]:
    """Load a CSV with image/path and text/label columns.

    Paths are resolved relative to the manifest. A one-column CSV is also
    accepted: its first value is the image filename and the stem is its label.
    Optional metadata columns include source, split, writer_id, document_id,
    group_id, sample_type, and raw_text.
    """
    manifest = Path(path)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []

    header = [value.strip().lower() for value in rows[0]]
    path_idx = _column_index(header, _PATH_COLUMNS)
    text_idx = _column_index(header, _TEXT_COLUMNS)
    has_header = path_idx is not None and text_idx is not None
    if has_header:
        start = 1
    elif path_idx is not None and len(rows[0]) == 1:
        # A one-column image list may include only an image/path header; labels
        # in this legacy format are the image filename stems.
        has_header = True
        start = 1
    else:
        start = 0
        path_idx = 0
        text_idx = 1 if len(rows[0]) > 1 else None

    metadata_indices = {
        name: _column_index(header, {name}) if has_header else None
        for name in (
            "source",
            "split",
            "writer_id",
            "document_id",
            "group_id",
            "sample_type",
            "raw_text",
            "generation_kind",
            "font_name",
            "font_license",
        )
    }
    samples: list[Sample] = []
    for row in rows[start:]:
        image_value = _cell(row, path_idx)
        if not image_value:
            continue
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        text = _cell(row, text_idx) if text_idx is not None else image_path.stem
        # Only infer a label from a filename in the documented one-column case.
        if text_idx is None:
            text = image_path.stem
        raw_text = _cell(row, metadata_indices["raw_text"]) or text
        samples.append(
            Sample(
                image_path=image_path,
                text=text,
                source=_cell(row, metadata_indices["source"]),
                split=_cell(row, metadata_indices["split"]),
                writer_id=_cell(row, metadata_indices["writer_id"]),
                document_id=_cell(row, metadata_indices["document_id"]),
                group_id=_cell(row, metadata_indices["group_id"]),
                sample_type=_cell(row, metadata_indices["sample_type"]) or "real",
                raw_text=raw_text,
                generation_kind=_cell(row, metadata_indices["generation_kind"]),
                font_name=_cell(row, metadata_indices["font_name"]),
                font_license=_cell(row, metadata_indices["font_license"]),
            )
        )
    return samples


def discover_images(root: str | Path) -> list[Sample]:
    """Discover image-folder datasets; image filename stems are the labels."""
    base = Path(root)
    if not base.exists():
        return []
    return [
        Sample(path, path.stem, raw_text=path.stem)
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def validate_samples(samples: list[Sample], *, check_files: bool = True) -> list[str]:
    """Return human-readable data issues; an empty list means valid."""
    issues: list[str] = []
    if not samples:
        issues.append("No labeled images found.")
    for sample in samples:
        if not sample.text.strip():
            issues.append(f"Empty label: {sample.image_path}")
        if check_files and not sample.image_path.is_file():
            issues.append(f"Image file not found: {sample.image_path}")
    return issues


def normalize_text(text: str, variants: dict[str, str] | None = None) -> str:
    """Apply Unicode, case, and whitespace normalization plus exact variants.

    Variant keys and values use the same normalization, so a mapping can
    standardize known spelling variants without changing unrelated words.
    """
    value = " ".join(unicodedata.normalize("NFKC", text).split()).upper()
    if not variants:
        return value
    normalized_variants = {
        " ".join(unicodedata.normalize("NFKC", key).split()).upper():
        " ".join(unicodedata.normalize("NFKC", target).split()).upper()
        for key, target in variants.items()
    }
    seen: set[str] = set()
    while value in normalized_variants and value not in seen:
        seen.add(value)
        value = normalized_variants[value]
    return value


def image_sha256(path: str | Path) -> str:
    """Hash image bytes so exact copies stay in one split."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_samples(samples: list[Sample]) -> tuple[list[Sample], int]:
    """Drop exact image duplicates; reject conflicting labels for identical bytes."""
    by_hash: dict[str, Sample] = {}
    unique: list[Sample] = []
    duplicates = 0
    for sample in samples:
        if not sample.image_path.is_file():
            raise FileNotFoundError(sample.image_path)
        digest = image_sha256(sample.image_path)
        previous = by_hash.get(digest)
        if previous is not None:
            if normalize_text(previous.text) != normalize_text(sample.text):
                raise ValueError(
                    "Identical image bytes have conflicting labels: "
                    f"{previous.image_path} ({previous.text!r}) and "
                    f"{sample.image_path} ({sample.text!r})"
                )
            duplicates += 1
            continue
        by_hash[digest] = sample
        unique.append(sample)
    return unique, duplicates


def _group_tokens(sample: Sample) -> list[str]:
    source = sample.source.strip() or "unknown-source"
    tokens = [f"image:{image_sha256(sample.image_path)}"]
    for field in ("writer_id", "document_id", "group_id"):
        value = getattr(sample, field).strip()
        if value:
            tokens.append(f"{source}:{field}:{value}")
    return tokens


def split_samples(
    samples: list[Sample],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Make deterministic train/val/test splits with image and ID group isolation.

    Images sharing a writer, document, explicit group ID, or exact byte hash
    are connected into one split. Without writer/document metadata, the function
    can prevent exact-image leakage but cannot guarantee writer isolation.
    """
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values for train/val/test")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("train/val/test fractions must sum to 1")
    if len(samples) < 3:
        raise ValueError("At least three samples are needed for train/val/test splits")
    for issue in validate_samples(samples):
        raise ValueError(issue)

    # Union samples that share any supported leakage key. This also handles
    # the case where writer and document IDs form overlapping groups.
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    first_for_token: dict[str, int] = {}
    for index, sample in enumerate(samples):
        for token in _group_tokens(sample):
            prior = first_for_token.setdefault(token, index)
            union(index, prior)

    groups: dict[int, list[int]] = {}
    for index in range(len(samples)):
        groups.setdefault(find(index), []).append(index)
    if len(groups) < 3:
        raise ValueError(f"Only {len(groups)} independent groups; cannot populate all three splits")

    names = ("train", "val", "test")
    targets = [len(samples) * value for value in fractions]
    counts = [0, 0, 0]
    rng = random.Random(seed)
    group_indices = list(groups.values())
    rng.shuffle(group_indices)
    group_indices.sort(key=len, reverse=True)
    assignments: dict[int, str] = {}
    for group in group_indices:
        # Prefer the split furthest below its desired share. Random tie values
        # are deterministic for a fixed seed and avoid a fixed split bias.
        tie_breaks = [rng.random() for _ in names]
        selected = min(
            range(3),
            key=lambda split_index: (
                counts[split_index] / max(targets[split_index], 1.0),
                tie_breaks[split_index],
            ),
        )
        for index in group:
            assignments[index] = names[selected]
        counts[selected] += len(group)

    result = {name: [] for name in names}
    for index, sample in enumerate(samples):
        result[assignments[index]].append(sample._replace(split=assignments[index]))
    return result


def assert_no_leakage(train_samples: list[Sample], validation_samples: list[Sample]) -> None:
    """Reject exact-image or supplied metadata-group overlap across two splits."""
    train_hashes = {image_sha256(sample.image_path) for sample in train_samples}
    validation_hashes = {image_sha256(sample.image_path) for sample in validation_samples}
    common_hashes = train_hashes & validation_hashes
    if common_hashes:
        raise ValueError(f"Train/validation contain {len(common_hashes)} identical image files")
    for field in ("writer_id", "document_id", "group_id"):
        train_ids = {
            (sample.source.strip(), getattr(sample, field).strip())
            for sample in train_samples
            if getattr(sample, field).strip()
        }
        validation_ids = {
            (sample.source.strip(), getattr(sample, field).strip())
            for sample in validation_samples
            if getattr(sample, field).strip()
        }
        common_ids = train_ids & validation_ids
        if common_ids:
            raise ValueError(f"{len(common_ids)} {field} values appear in both train and validation")
