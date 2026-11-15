"""Render configurable synthetic medicine/dosage lines with licensed fonts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True, help="One manually reviewed medicine name per line")
    parser.add_argument("--font", type=Path, action="append", required=True, help="Licensed handwriting font file; repeat as needed")
    parser.add_argument("--font-license", action="append", required=True, help="License label for each --font; repeat in the same order")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--medicine-only", action="store_true", help="Do not add dosage/frequency patterns")
    args = parser.parse_args()
    names = [line.strip() for line in args.vocabulary.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    manifest = generate_synthetic_dataset(
        names,
        args.font,
        args.output,
        font_licenses=args.font_license,
        count=args.count,
        seed=args.seed,
        include_instructions=not args.medicine_only,
    )
    print(f"Generated {args.count} synthetic images; manifest: {manifest}")
    print("Synthetic samples are marked as synthetic. Never report them as real prescription data.")


if __name__ == "__main__":
    main()
