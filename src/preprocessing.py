"""Image preparation and reproducible augmentation for handwritten word crops."""

from __future__ import annotations

import random
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_image(source: str | Path | BinaryIO | Image.Image) -> Image.Image:
    """Load an image safely and return an RGB copy."""
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        image = Image.open(source)
    return ImageOps.exif_transpose(image).convert("RGB")


def _deskew_score(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L"), dtype=np.int16)
    ink = np.where(pixels < 200, 255 - pixels, 0)
    projection = ink.sum(axis=1, dtype=np.int64)
    return int(np.square(projection).sum(dtype=np.int64))


def deskew_image(image: Image.Image, *, max_angle: float = 5.0, step: float = 1.0) -> Image.Image:
    """Estimate a small baseline angle using horizontal ink projections."""
    if max_angle < 0 or step <= 0:
        raise ValueError("max_angle must be non-negative and step must be positive")
    source = image.convert("L")
    if max_angle == 0 or source.width < 12 or source.height < 12:
        return source
    best = source
    best_score = _deskew_score(source)
    count = int(round(2 * max_angle / step))
    for index in range(count + 1):
        angle = -max_angle + index * step
        if abs(angle) < 1e-9:
            continue
        rotated = source.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        score = _deskew_score(rotated)
        if score > best_score:
            best, best_score = rotated, score
    return best


def preprocess_image(
    source: str | Path | BinaryIO | Image.Image,
    image_size: tuple[int, int] = (384, 128),
    *,
    invert: bool = False,
    deskew: bool = True,
) -> Image.Image:
    """Normalize a crop, preserve its aspect ratio, and pad on white.

    ``image_size`` is (width, height). The crop is centered vertically and
    left-aligned to retain natural reading order. Set the dimensions to a square
    for TrOCR, whose vision encoder expects square pixels.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size dimensions must be positive")
    image = load_image(source).convert("L")
    image = ImageOps.autocontrast(image)
    if deskew:
        image = deskew_image(image)
    if invert:
        image = ImageOps.invert(image)
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), color=255)
    y = (height - image.height) // 2
    canvas.paste(image, (0, y))
    return canvas


def _elastic_distortion(image: Image.Image, rng: random.Random, magnitude: float = 3.0) -> Image.Image:
    width, height = image.size
    grid = 4
    xs = [round(index * width / grid) for index in range(grid + 1)]
    ys = [round(index * height / grid) for index in range(grid + 1)]
    points: dict[tuple[int, int], tuple[float, float]] = {}
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            edge = row in (0, grid) or column in (0, grid)
            dx = 0.0 if edge else rng.uniform(-magnitude, magnitude)
            dy = 0.0 if edge else rng.uniform(-magnitude, magnitude)
            points[(column, row)] = (x + dx, y + dy)

    mesh = []
    for row in range(grid):
        for column in range(grid):
            p00 = points[(column, row)]
            p10 = points[(column + 1, row)]
            p11 = points[(column + 1, row + 1)]
            p01 = points[(column, row + 1)]
            quad = (*p00, *p10, *p11, *p01)
            mesh.append(((xs[column], ys[row], xs[column + 1], ys[row + 1]), quad))
    return image.transform(
        image.size,
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )


def augment_image(
    image: Image.Image,
    *,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Apply handwriting-like rotation, blur, noise, paper, and elastic warps."""
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    generator = rng or random.Random(seed)
    result = image.convert("L")

    if generator.random() < 0.65:
        result = result.rotate(
            generator.uniform(-4.0, 4.0),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=255,
        )
    if generator.random() < 0.25:
        result = result.filter(ImageFilter.GaussianBlur(generator.uniform(0.2, 1.0)))
    if generator.random() < 0.30:
        result = _elastic_distortion(result, generator)

    if generator.random() < 0.25:
        noise_rng = np.random.default_rng(generator.getrandbits(64))
        pixels = np.asarray(result, dtype=np.uint8)
        texture = np.clip(noise_rng.normal(249, 2.5, size=pixels.shape), 232, 255).astype(np.uint8)
        result = Image.fromarray(np.where(pixels > 205, texture, pixels).astype(np.uint8))

    if generator.random() < 0.30:
        noise_rng = np.random.default_rng(generator.getrandbits(64))
        pixels = np.asarray(result, dtype=np.int16)
        noisy = np.clip(pixels + noise_rng.normal(0, 7, size=pixels.shape), 0, 255).astype(np.uint8)
        result = Image.fromarray(noisy)
    return result
