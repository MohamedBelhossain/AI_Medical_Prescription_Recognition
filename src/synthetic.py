"""Synthetic word/line image creation from user-supplied labels and licensed fonts."""

from __future__ import annotations

import random
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .preprocessing import augment_image, preprocess_image

DOSAGE_TOKENS = ("250 MG", "500 MG", "5 ML", "10 ML", "1+0+1", "0+1+0")
FREQUENCY_TOKENS = ("OD", "BD", "TDS", "QID")


def _synthetic_text(medicine: str, rng: random.Random, include_instructions: bool) -> tuple[str, str]:
    if not include_instructions or rng.random() < 0.45:
        return medicine, "medicine"
    dosage = rng.choice(DOSAGE_TOKENS)
    frequency = rng.choice(FREQUENCY_TOKENS)
    style = rng.choice(("medicine+dosage", "medicine+frequency", "medicine+dosage+frequency", "dosage+frequency"))
    if style == "medicine+dosage":
        return f"{medicine} {dosage}", style
    if style == "medicine+frequency":
        return f"{medicine} {frequency}", style
    if style == "medicine+dosage+frequency":
        return f"{medicine} {dosage} {frequency}", style
    return f"{dosage} {frequency}", style


def generate_synthetic_dataset(
    medicine_names: list[str],
    font_paths: list[str | Path],
    output_dir: str | Path,
    *,
    font_licenses: list[str],
    count: int,
    seed: int = 42,
    include_instructions: bool = True,
    image_size: tuple[int, int] = (384, 384),
) -> Path:
    """Generate images and a manifest marked as synthetic; never used as real data."""
    if count < 1:
        raise ValueError("count must be positive")
    names = [" ".join(name.split()).upper() for name in medicine_names if name.strip()]
    if not names:
        raise ValueError("provide at least one medicine name in the input vocabulary")
    if not font_paths:
        raise ValueError("provide at least one licensed handwriting font with --font")
    if len(font_licenses) != len(font_paths) or any(not license_name.strip() for license_name in font_licenses):
        raise ValueError("provide one non-empty font license label for each --font")
    fonts = [Path(path) for path in font_paths]
    for path in fonts:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[tuple[str, str, str, str, str, str]] = []
    width, height = image_size

    for index in range(count):
        medicine = rng.choice(names)
        text, kind = _synthetic_text(medicine, rng, include_instructions)
        font_index = rng.randrange(len(fonts))
        font_path = fonts[font_index]
        font_size = rng.randint(max(18, height // 14), max(22, height // 7))
        while True:
            font = ImageFont.truetype(str(font_path), size=font_size)
            measure = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
            text_width, text_height = measure[2] - measure[0], measure[3] - measure[1]
            if (text_width <= width - 8 and text_height <= height // 2) or font_size <= 12:
                break
            font_size -= 2
        canvas = Image.new("L", image_size, color=255)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        x = rng.randint(0, max(0, width - text_width - 4))
        y = rng.randint(max(0, height // 3 - 12), max(height // 3 - 11, height - text_height - 4))
        draw.text((x - bbox[0], y - bbox[1]), text, fill=rng.randint(0, 65), font=font)
        crop = preprocess_image(canvas, image_size, deskew=False)
        crop = augment_image(crop, rng=rng)
        filename = f"synthetic_{index:07d}.png"
        crop.save(images_dir / filename)
        rows.append((f"images/{filename}", text, kind, font_path.name, font_licenses[font_index], "synthetic"))

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(("image_path", "text", "source", "split", "sample_type", "generation_kind", "font_name", "font_license"))
        for image_path, text, kind, font_name, font_license, sample_type in rows:
            writer.writerow((image_path, text, "synthetic", "", sample_type, kind, font_name, font_license))
    generation_config = {
        "seed": seed,
        "count": count,
        "image_size": list(image_size),
        "include_instructions": include_instructions,
        "medicine_name_count": len(names),
        "medicine_names_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "fonts": [
            {"name": path.name, "license": license_name}
            for path, license_name in zip(fonts, font_licenses)
        ],
    }
    (root / "generation_config.json").write_text(json.dumps(generation_config, indent=2) + "\n", encoding="utf-8")
    return manifest

