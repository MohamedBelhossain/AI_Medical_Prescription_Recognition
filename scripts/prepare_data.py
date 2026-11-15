"""Normalize input manifests and make writer/document-aware train/val/test splits.

Example:
    python scripts/prepare_data.py \
      --input doctor-bd=data/raw/doctor-bd/labels.csv \
      --input rxhandbd=data/raw/rxhandbd/train_labels.csv \
      --input rxhandbd=data/raw/rxhandbd/test_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import Sample, deduplicate_samples, load_manifest, normalize_text, split_samples, validate_samples
from src.reproducibility import seed_everything

FIELDS = (
    "image_path",
    "text",
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


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be SOURCE=CSV_PATH")
    source, path = value.split("=", 1)
    if not source.strip() or not path.strip():
        raise argparse.ArgumentTypeError("input must include a source name and CSV path")
    return source.strip(), Path(path)


def load_variants(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}
    header = [value.strip().lower() for value in rows[0]]
    if "variant" in header and "canonical" in header:
        variant_idx, canonical_idx, start = header.index("variant"), header.index("canonical"), 1
    else:
        if len(rows[0]) < 2:
            raise ValueError("variant CSV needs variant,canonical columns")
        variant_idx, canonical_idx, start = 0, 1, 0
    return {
        row[variant_idx].strip(): row[canonical_idx].strip()
        for row in rows[start:]
        if len(row) > max(variant_idx, canonical_idx) and row[variant_idx].strip() and row[canonical_idx].strip()
    }


def write_manifest(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in samples:
            relative = os.path.relpath(sample.image_path.resolve(), path.parent.resolve())
            writer.writerow(
                {
                    "image_path": Path(relative).as_posix(),
                    "text": sample.text,
                    "source": sample.source,
                    "split": sample.split,
                    "writer_id": sample.writer_id,
                    "document_id": sample.document_id,
                    "group_id": sample.group_id,
                    "sample_type": sample.sample_type,
                    "raw_text": sample.raw_text,
                    "generation_kind": sample.generation_kind,
                    "font_name": sample.font_name,
                    "font_license": sample.font_license,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True, metavar="SOURCE=CSV")
    parser.add_argument("--variants", type=Path, help="Optional CSV with variant,canonical columns")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    fractions = (args.train_fraction, args.val_fraction, args.test_fraction)
    variants = load_variants(args.variants)
    samples: list[Sample] = []
    for source, manifest in args.input:
        for sample in load_manifest(manifest):
            raw_text = sample.raw_text or sample.text
            samples.append(
                sample._replace(
                    text=normalize_text(sample.text, variants),
                    raw_text=raw_text,
                    source=sample.source or source,
                    sample_type=sample.sample_type or "real",
                )
            )
    issues = validate_samples(samples)
    if issues:
        raise SystemExit("Invalid input manifest:\n" + "\n".join(issues[:25]))
    samples, duplicate_count = deduplicate_samples(samples)
    split = split_samples(samples, fractions=fractions, seed=args.seed)
    unified = [sample for split_name in ("train", "val", "test") for sample in split[split_name]]
    write_manifest(args.output, unified)
    for split_name in ("train", "val", "test"):
        write_manifest(args.output.with_name(f"{split_name}.csv"), split[split_name])
    config = {
        "seed": args.seed,
        "fractions": {"train": fractions[0], "val": fractions[1], "test": fractions[2]},
        "inputs": [{"source": source, "manifest": str(path.resolve())} for source, path in args.input],
        "variants_file": str(args.variants.resolve()) if args.variants else None,
        "exact_duplicates_removed": duplicate_count,
        "unique_samples": len(samples),
        "writer_document_grouping_available": any(
            sample.writer_id or sample.document_id or sample.group_id for sample in samples
        ),
        "split_method": "deterministic greedy assignment of connected writer/document/group/image-hash components",
    }
    args.output.with_name("prepare_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"Unique images: {len(samples)} (removed {duplicate_count} exact duplicates)")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {len(split[split_name])} samples")
    if not any(sample.writer_id or sample.document_id or sample.group_id for sample in samples):
        print("Note: writer/document/group IDs were absent; only exact-image duplicates were kept together.")
    print(f"Wrote manifests beside {args.output}")


if __name__ == "__main__":
    main()
