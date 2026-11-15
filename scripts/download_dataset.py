"""Download the small public prescription handwriting sample dataset."""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

REPO_ID = "avi-kai/Medical_Prescription_Handwritten_Words"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main?recursive=true&expand=false"
RAW_DIR = Path(__file__).resolve().parents[1] / "dataset" / "raw" / "images"
MANIFEST = Path(__file__).resolve().parents[1] / "dataset" / "labels" / "manifest.csv"


def main() -> None:
    request = urllib.request.Request(API_URL, headers={"User-Agent": "prescription-ocr-project/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.load(response)
    files = [entry["path"] for entry in entries if entry.get("type") == "file" and "/images/" in entry["path"] and entry["path"].lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        raise RuntimeError("No supported image files found in the dataset repository")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for remote_path in files:
        filename = Path(remote_path).name
        destination = RAW_DIR / filename
        if not destination.exists():
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{remote_path}"
            request = urllib.request.Request(url, headers={"User-Agent": "prescription-ocr-project/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
                output.write(response.read())
        samples.append((f"../raw/images/{filename}", Path(filename).stem))
        print(f"Downloaded {filename}")
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "text"])
        writer.writerows(samples)
    print(f"Saved {len(samples)} samples to {MANIFEST}")


if __name__ == "__main__":
    main()
