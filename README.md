# Handwritten Prescription Word Recognition

A research starter for recognizing medicine names from **cropped handwritten words or lines**. Phase 1 uses pretrained TrOCR, supports optional fine-tuning on manually labeled crops, and can rank OCR text against a reviewed medicine vocabulary. It is not a clinical system.

> **Safety:** This is an assistive research demo, not a replacement for a pharmacist or doctor. OCR output and vocabulary matches are fallible and must be checked by a qualified person. Never use the result as a medication instruction or medical decision.

## Setup

Python 3.11–3.14 is supported by the pinned project dependencies in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The first TrOCR run downloads model weights from Hugging Face. Inference runs locally after the model is downloaded.

The existing 46-image public sample can be recreated with:

```bash
python scripts/download_dataset.py
```

## Current project contents

- `app/app.py`: Streamlit single-crop OCR demo with optional lexicon candidates, an editable result, and a human confirmation button.
- `src/dataset.py`: manifest loading, label normalization, image hashing, duplicate checks, and grouped splitting.
- `src/preprocessing.py`: aspect-preserving resize/padding, contrast normalization, deskewing, and handwriting-style augmentation.
- `src/model.py`, `scripts/train_crnn.py`: existing CRNN/CTC comparison baseline.
- `scripts/prepare_data.py`: unify CSV manifests and write train/validation/test manifests.
- `scripts/train_trocr.py`: fine-tune `microsoft/trocr-base-handwritten` and save the best checkpoint, config, and history.
- `src/synthetic.py`, `scripts/generate_synthetic.py`: make clearly marked synthetic word/line samples using a user vocabulary and licensed font files.
- `src/postprocessing.py`: top-k vocabulary matching with normalized edit-similarity scores.
- `scripts/evaluate.py`, `src/metrics.py`: CER, WER, exact-match, optional top-k accuracy, real/synthetic groups, rare/frequent groups, and error examples.
- `DATA_SOURCES.md`: dataset access and license notes, plus the papers used to shape the pipeline.
- `tests/`: tests for data, preprocessing, splitting, post-correction, and metrics.

The five existing notebooks are kept in place. Four are empty placeholders and `01_exploration.ipynb` has no analysis cells yet; the executable workflows live in scripts so they can be run from the command line.

## Data and privacy

Download datasets manually, review their terms, and keep extracted files under the ignored `data/` directory. Do not commit real prescriptions, patient details, or derived images. The existing 46-image public sample is in `dataset/raw/images/` on this checkout; these images are ignored by Git and are too small for meaningful training or performance claims.

For each downloaded CSV, keep image paths relative to that CSV and include `image_path` (or `image`/`path`/`filename`) plus `text` (or `label`/`transcription`/`medicine_name`). Optional metadata columns are `source`, `split`, `writer_id`, `document_id`, `group_id`, `sample_type`, and `raw_text`. If a dataset provides writer or prescription IDs, retain them.

Example:

```bash
python scripts/prepare_data.py \
  --input doctor-bd=data/raw/doctor-bd/labels.csv \
  --input rxhandbd=data/raw/rxhandbd/train_labels.csv \
  --input rxhandbd=data/raw/rxhandbd/test_labels.csv \
  --output data/processed/manifest.csv \
  --seed 42
```

The command normalizes case/spacing, drops exact image duplicates (and rejects conflicting labels), and creates `manifest.csv`, `train.csv`, `val.csv`, and `test.csv`. Samples sharing a provided writer, document, or group ID stay together. Exact copies also stay together. When a dataset has no writer/document metadata, writer-level isolation cannot be guaranteed; the script prints this limitation.

Optional spelling variants use a CSV with `variant,canonical` columns. Keep brand names separate unless the reviewed annotation explicitly says they are equivalent; normalization must not silently map different products to one generic.

## Fine-tune TrOCR

```bash
python scripts/train_trocr.py \
  --train-manifest data/processed/train.csv \
  --val-manifest data/processed/val.csv \
  --config configs/trocr.json \
  --output models/trocr \
  --seed 42
```

The script checks train/validation image and metadata-group overlap, applies augmentation to training images only, evaluates after each epoch, and saves the best validation-CER checkpoint under `models/trocr/best/`. The effective hyperparameters and epoch history are saved alongside it. The project also retains the CRNN/CTC trainer for an optional comparison.

```bash
python scripts/train_crnn.py \
  --train-manifest data/processed/train.csv \
  --val-manifest data/processed/val.csv \
  --epochs 30 \
  --output models/crnn.pt
```

## Synthetic training samples

Provide a reviewed medicine vocabulary and one or more handwriting fonts whose licenses permit the intended use. The script does not download fonts or drug lists.

```bash
python scripts/generate_synthetic.py \
  --vocabulary data/vocabulary/medicine_names.txt \
  --font data/fonts/Handwriting.ttf \
  --font-license OFL-1.1 \
  --count 1000 \
  --seed 42 \
  --output data/synthetic
```

The manifest marks each generated sample as `source=synthetic` and `sample_type=synthetic`, and records the font name and license label. Generated images are not real prescriptions. Keep them separate in reporting and never use synthetic-only metrics as evidence of real-world accuracy.

## Evaluate

```bash
python scripts/evaluate.py \
  --manifest data/processed/test.csv \
  --train-manifest data/processed/train.csv \
  --model models/trocr/best \
  --lexicon data/vocabulary/medicine_names.txt \
  --output-dir results/test
```

This writes `metrics.json`, `predictions.csv`, and `errors.csv`. Similarity scores are string-matching scores, not calibrated confidence probabilities. Frequency groups use counts in the training manifest; the default rare threshold is fewer than five training examples.

## Demo

```bash
streamlit run app/app.py
```

Upload one cropped handwritten word or line, optionally upload a reviewed one-name-per-line lexicon, and edit/confirm the output. The demo does not save the uploaded image or confirmed text. Do not upload identifiable patient data to a shared deployment.

## Tests and smoke checks

```bash
python -m pytest -q
```

Tests cover the local utilities; they do not train a model or establish clinical accuracy. No model metrics are claimed until a held-out real dataset has been prepared and evaluated.

## Phase 2

Full-page extraction of medicine, dosage, and frequency with a vision-language model is optional and has not been implemented. It requires a suitable page-level dataset, privacy review, and substantial compute. Phase 1 is intentionally limited to cropped word/line recognition.
