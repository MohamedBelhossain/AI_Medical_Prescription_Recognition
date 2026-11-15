"""Run OCR on a held-out manifest and save metrics, predictions, and errors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest, normalize_text, validate_samples
from src.metrics import evaluate_rows
from src.postprocessing import load_lexicon, rank_candidates
from src.prediction import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Held-out test CSV")
    parser.add_argument("--train-manifest", type=Path, help="Training CSV for rare/frequent label grouping")
    parser.add_argument("--model", default="microsoft/trocr-base-handwritten", help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--lexicon", type=Path, help="One reviewed medicine candidate per line")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rare-below", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "evaluation")
    args = parser.parse_args()
    if args.top_k < 1 or args.rare_below < 1:
        raise SystemExit("top-k and rare-below must be positive")

    samples = load_manifest(args.manifest)
    issues = validate_samples(samples)
    if issues:
        raise SystemExit("Invalid evaluation manifest:\n" + "\n".join(issues[:20]))
    training_counts: Counter[str] | None = None
    if args.train_manifest:
        train_samples = load_manifest(args.train_manifest)
        training_counts = Counter(normalize_text(sample.text) for sample in train_samples)
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None

    prediction_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        raw_prediction = predict_image(sample.image_path, model_name=args.model)
        prediction = normalize_text(raw_prediction)
        candidates = rank_candidates(prediction, lexicon, top_k=args.top_k) if lexicon else []
        prediction_rows.append(
            {
                "image_path": str(sample.image_path),
                "reference": normalize_text(sample.text),
                "prediction": prediction,
                "source": sample.source,
                "sample_type": sample.sample_type or "real",
                "candidates": json.dumps(candidates, ensure_ascii=False),
            }
        )
        if prediction != normalize_text(sample.text):
            errors.append(prediction_rows[-1])
        if index % 25 == 0 or index == len(samples):
            print(f"Predicted {index}/{len(samples)}")

    metrics = evaluate_rows(
        prediction_rows,
        training_label_counts=training_counts,
        rare_below=args.rare_below,
        lexicon=lexicon,
        top_ks=tuple(sorted(set((1, 3, args.top_k)))),
    )
    metrics["model"] = args.model
    metrics["evaluation_manifest"] = str(args.manifest.resolve())
    metrics["training_manifest"] = str(args.train_manifest.resolve()) if args.train_manifest else None
    metrics["lexicon"] = str(args.lexicon.resolve()) if args.lexicon else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("predictions.csv", prediction_rows), ("errors.csv", errors)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_path", "reference", "prediction", "source", "sample_type", "candidates"))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics and examples to {args.output_dir}")


if __name__ == "__main__":
    main()
