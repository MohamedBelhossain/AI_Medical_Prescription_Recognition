"""Train the CRNN/CTC baseline on labeled handwritten word crops.

Example:
    python scripts/train_crnn.py \
      --train-manifest dataset/labels/train.csv \
      --val-manifest dataset/labels/val.csv \
      --epochs 30 --output models/crnn.pt

Each CSV must contain ``image,text`` columns. Image paths are relative to the
CSV file. Class 0 is reserved for the CTC blank token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import assert_no_leakage, load_manifest, validate_samples
from src.model import CRNN
from src.preprocessing import preprocess_image
from src.prediction import greedy_ctc_decode
from src.reproducibility import seed_everything


IMAGE_SIZE = (384, 128)


class WordDataset(Dataset):
    def __init__(self, manifest: Path, alphabet: str):
        self.samples = load_manifest(manifest)
        issues = validate_samples(self.samples)
        if issues:
            raise ValueError("Invalid dataset:\n" + "\n".join(issues[:20]))
        self.char_to_id = {char: index + 1 for index, char in enumerate(alphabet)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        unknown = sorted(set(sample.text) - self.char_to_id.keys())
        if unknown:
            raise ValueError(f"Label contains characters absent from training alphabet: {unknown!r}")
        image = preprocess_image(sample.image_path, IMAGE_SIZE)
        pixels = torch.from_numpy(np.array(image, copy=True)).float().div_(255).unsqueeze(0)
        pixels = (pixels - 0.5) / 0.5
        target = torch.tensor([self.char_to_id[char] for char in sample.text], dtype=torch.long)
        return pixels, target, sample.text


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for col, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def collate_batch(items):
    images, targets, texts = zip(*items)
    return torch.stack(images), torch.cat(targets), torch.tensor([len(target) for target in targets]), list(texts)


def run_epoch(model, loader, optimizer, criterion, device, alphabet, train: bool):
    model.train(train)
    total_loss = 0.0
    predictions: list[str] = []
    references: list[str] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets, target_lengths, texts in loader:
            images, targets = images.to(device), targets.to(device)
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device="cpu")
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            predictions.extend(greedy_ctc_decode(log_probs, alphabet))
            references.extend(texts)
    chars = sum(len(text) for text in references)
    errors = sum(edit_distance(ref, hyp) for ref, hyp in zip(references, predictions))
    word_accuracy = sum(ref == hyp for ref, hyp in zip(references, predictions)) / max(len(references), 1)
    return total_loss / max(len(loader.dataset), 1), errors / max(chars, 1), word_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "crnn.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    seed_everything(args.seed)
    alphabet = "".join(sorted({char for sample in load_manifest(args.train_manifest) for char in sample.text}))
    if not alphabet:
        raise SystemExit("Training manifest has no labeled samples")
    train_set = WordDataset(args.train_manifest, alphabet)
    val_set = WordDataset(args.val_manifest, alphabet)
    if not val_set:
        raise SystemExit("Validation manifest has no samples")
    assert_no_leakage(train_set.samples, val_set.samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = CRNN(num_classes=len(alphabet) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    training_config = {
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.val_manifest.resolve()),
        "output": str(args.output.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "alphabet": alphabet,
        "image_size": IMAGE_SIZE,
        "device": str(device),
    }
    args.output.with_name("training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Training samples: {len(train_set)} | validation samples: {len(val_set)} | device: {device}")
    print(f"Alphabet size: {len(alphabet)} | image size: {IMAGE_SIZE}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_cer, _ = run_epoch(model, train_loader, optimizer, criterion, device, alphabet, True)
        val_loss, val_cer, val_word_accuracy = run_epoch(model, val_loader, optimizer, criterion, device, alphabet, False)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_CER={train_cer:.3f} "
            f"val_loss={val_loss:.4f} val_CER={val_cer:.3f} val_word_accuracy={val_word_accuracy:.3f}"
        )
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "alphabet": alphabet,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "validation_cer": val_cer,
                    "validation_word_accuracy": val_word_accuracy,
                    "training_config": training_config,
                },
                args.output,
            )
    print(f"Best model saved to {args.output} (validation CER {best_cer:.3f})")


if __name__ == "__main__":
    main()
