"""Fine-tune TrOCR on cropped word/line images using train/validation CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import assert_no_leakage, load_manifest, normalize_text, validate_samples
from src.metrics import edit_distance
from src.preprocessing import augment_image, preprocess_image
from src.reproducibility import seed_everything


class WordDataset(Dataset):
    def __init__(self, manifest: Path, processor, max_target_length: int, *, augment: bool, seed: int):
        self.samples = load_manifest(manifest)
        issues = validate_samples(self.samples)
        if issues:
            raise ValueError("Invalid dataset:\n" + "\n".join(issues[:20]))
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = preprocess_image(sample.image_path, (384, 384)).convert("RGB")
        if self.augment:
            image = augment_image(image, seed=self.seed + self.epoch * len(self) + index).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        text = normalize_text(sample.text)
        labels = self.processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        return pixels, torch.tensor(labels, dtype=torch.long), text


def collate_batch(items):
    pixels, label_sequences, texts = zip(*items)
    max_length = max(sequence.numel() for sequence in label_sequences)
    labels = torch.full((len(items), max_length), fill_value=-100, dtype=torch.long)
    for row, sequence in enumerate(label_sequences):
        labels[row, : sequence.numel()] = sequence
    return torch.stack(pixels), labels, list(texts)


def run_epoch(model, loader, optimizer, device, tokenizer, *, train: bool, max_target_length: int):
    model.train(train)
    total_loss = 0.0
    references: list[str] = []
    hypotheses: list[str] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for pixel_values, labels, texts in loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            output = model(pixel_values=pixel_values, labels=labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(output.loss.detach().item()) * len(texts)
            if not train:
                generated = model.generate(pixel_values, max_length=max_target_length, num_beams=1)
                hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
                references.extend(texts)

    average_loss = total_loss / max(len(loader.dataset), 1)
    if train:
        return average_loss, None, None, None
    char_count = sum(len(text) for text in references)
    word_count = sum(len(text.split()) for text in references)
    char_errors = sum(edit_distance(ref, hyp.strip()) for ref, hyp in zip(references, hypotheses))
    word_errors = sum(edit_distance(ref.split(), hyp.strip().split()) for ref, hyp in zip(references, hypotheses))
    exact = sum(ref == hyp.strip() for ref, hyp in zip(references, hypotheses)) / max(len(references), 1)
    return average_loss, char_errors / max(char_count, 1), word_errors / max(word_count, 1), exact


def parse_args():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=ROOT / "configs" / "trocr.json")
    known, _ = preliminary.parse_known_args()
    config = json.loads(known.config.read_text(encoding="utf-8")) if known.config.is_file() else {}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--model", default=config.get("model", "microsoft/trocr-base-handwritten"))
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "trocr")
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 10))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 4))
    parser.add_argument("--learning-rate", type=float, default=config.get("learning_rate", 5e-5))
    parser.add_argument("--max-target-length", type=int, default=config.get("max_target_length", 64))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--device", default=config.get("device", "auto"))
    parser.add_argument("--no-augment", action="store_true", help="Disable on-the-fly training augmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.max_target_length) < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch-size, target length, and learning rate must be positive")
    seed_everything(args.seed)

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise SystemExit("Install the pinned project requirements to train TrOCR") from exc

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.max_length = args.max_target_length

    train_set = WordDataset(args.train_manifest, processor, args.max_target_length, augment=not args.no_augment, seed=args.seed)
    val_set = WordDataset(args.val_manifest, processor, args.max_target_length, augment=False, seed=args.seed)
    if not train_set or not val_set:
        raise SystemExit("Training and validation manifests must both contain samples")
    assert_no_leakage(train_set.samples, val_set.samples)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    collate = collate_batch
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    effective_config = {
        "model": args.model,
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.val_manifest.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_target_length": args.max_target_length,
        "seed": args.seed,
        "device": str(device),
        "augmentation": not args.no_augment,
    }
    (args.output / "training_config.json").write_text(json.dumps(effective_config, indent=2) + "\n", encoding="utf-8")
    (args.output / "history.json").write_text("[]\n", encoding="utf-8")
    history = []
    best_cer = float("inf")

    print(f"Train: {len(train_set)} | validation: {len(val_set)} | device: {device}")
    if not any(sample.writer_id or sample.document_id for sample in train_set.samples + val_set.samples):
        print("Writer/document IDs are absent; exact duplicate image leakage is still checked.")
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch - 1)
        train_loss, _, _, _ = run_epoch(model, train_loader, optimizer, device, tokenizer, train=True, max_target_length=args.max_target_length)
        val_loss, val_cer, val_wer, val_exact = run_epoch(model, val_loader, optimizer, device, tokenizer, train=False, max_target_length=args.max_target_length)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_cer": val_cer,
            "validation_wer": val_wer,
            "validation_exact_match_accuracy": val_exact,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} CER={val_cer:.4f} WER={val_wer:.4f} exact={val_exact:.4f}"
        )
        if val_cer < best_cer:
            best_cer = val_cer
            best_dir = args.output / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (args.output / "best_metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint saved under {args.output / 'best'} (validation CER {best_cer:.4f})")


if __name__ == "__main__":
    main()
