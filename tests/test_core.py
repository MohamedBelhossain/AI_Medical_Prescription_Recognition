import csv
from io import BytesIO
import shutil

import pytest
from PIL import Image

from src.dataset import Sample, assert_no_leakage, discover_images, load_manifest, normalize_text, split_samples, validate_samples
from src.metrics import edit_distance, evaluate_rows
from src.postprocessing import load_lexicon, rank_candidates
from src.preprocessing import augment_image, preprocess_image


def test_preprocess_preserves_image_and_pads(tmp_path):
    source = Image.new("RGB", (100, 40), "black")
    result = preprocess_image(source, (200, 60))
    assert result.mode == "L"
    assert result.size == (200, 60)
    assert source.mode == "RGB"  # preprocessing must not mutate caller input


def test_preprocess_rejects_invalid_size():
    with pytest.raises(ValueError):
        preprocess_image(Image.new("RGB", (5, 5)), (0, 10))


def test_preprocess_deskew_and_augmentation_preserve_expected_shape():
    source = Image.new("L", (120, 40), 255)
    result = preprocess_image(source, (200, 60))
    augmented = augment_image(result, seed=123)
    assert result.size == (200, 60)
    assert augmented.size == result.size
    assert augmented.tobytes() == augment_image(result, seed=123).tobytes()


def test_load_manifest_resolves_relative_paths_and_labels(tmp_path):
    (tmp_path / "images").mkdir()
    image = tmp_path / "images" / "drug.png"
    image.touch()
    manifest = tmp_path / "labels.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "text"])
        writer.writerow(["images/drug.png", "Drug Name"])
    samples = load_manifest(manifest)
    assert len(samples) == 1
    assert samples[0].image_path == image
    assert samples[0].text == "Drug Name"
    assert validate_samples(samples) == []


def test_load_manifest_supports_headerless_two_column_csv(tmp_path):
    manifest = tmp_path / "labels.csv"
    manifest.write_text("sample.png,Handwritten label\n", encoding="utf-8")
    samples = load_manifest(manifest)
    assert len(samples) == 1
    assert samples[0].image_path == tmp_path / "sample.png"
    assert samples[0].text == "Handwritten label"


def test_load_manifest_supports_one_column_image_header(tmp_path):
    (tmp_path / "Drug.png").touch()
    manifest = tmp_path / "images.csv"
    manifest.write_text("image_path\nDrug.png\n", encoding="utf-8")
    samples = load_manifest(manifest)
    assert [(sample.image_path.name, sample.text) for sample in samples] == [("Drug.png", "Drug")]


def test_load_manifest_preserves_group_and_sample_metadata(tmp_path):
    image = tmp_path / "sample.png"
    image.touch()
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "image_path,text,source,split,writer_id,document_id,sample_type\n"
        "sample.png,medicine,rxhandbd,train,w1,p3,real\n",
        encoding="utf-8",
    )
    sample = load_manifest(manifest)[0]
    assert sample.writer_id == "w1"
    assert sample.document_id == "p3"
    assert sample.split == "train"
    assert sample.sample_type == "real"


def test_blank_manifest_label_is_not_replaced_by_filename(tmp_path):
    manifest = tmp_path / "labels.csv"
    manifest.write_text("image,text\nunknown.png,\n", encoding="utf-8")
    samples = load_manifest(manifest)
    assert samples[0].text == ""
    assert "Empty label" in validate_samples(samples, check_files=False)[0]


def test_discover_images_uses_filename_as_label(tmp_path):
    (tmp_path / "Amoxicillin.png").touch()
    (tmp_path / "ignore.txt").touch()
    samples = discover_images(tmp_path)
    assert [(sample.image_path.name, sample.text) for sample in samples] == [("Amoxicillin.png", "Amoxicillin")]


def test_normalize_text_and_known_variant_mapping():
    assert normalize_text("  acetyl   SALICYLIC acid ") == "ACETYL SALICYLIC ACID"
    assert normalize_text("para-cetamol", {"para-cetamol": "paracetamol"}) == "PARACETAMOL"


def test_grouped_split_keeps_documents_together(tmp_path):
    samples = []
    for doc_index in range(6):
        for crop_index in range(2):
            image_path = tmp_path / f"{doc_index}-{crop_index}.png"
            Image.new("L", (12, 8), doc_index * 20 + crop_index).save(image_path)
            samples.append(
                Sample(
                    image_path,
                    f"DRUG {doc_index}",
                    source="test",
                    document_id=f"document-{doc_index}",
                )
            )
    split = split_samples(samples, seed=42)
    assert all(split[name] for name in ("train", "val", "test"))
    seen = {}
    for split_name, group in split.items():
        for sample in group:
            prior = seen.setdefault(sample.document_id, split_name)
            assert prior == split_name
            assert sample.split == split_name


def test_exact_image_copies_stay_together_across_sources(tmp_path):
    first_image = tmp_path / "first.png"
    duplicate_image = tmp_path / "duplicate.png"
    Image.new("L", (12, 8), 32).save(first_image)
    shutil.copyfile(first_image, duplicate_image)
    other_images = []
    for index in range(2):
        path = tmp_path / f"other-{index}.png"
        Image.new("L", (12, 8), 100 + index).save(path)
        other_images.append(path)
    samples = [
        Sample(first_image, "A", source="source-a"),
        Sample(duplicate_image, "A", source="source-b"),
        Sample(other_images[0], "B", source="source-a"),
        Sample(other_images[1], "C", source="source-b"),
    ]
    split = split_samples(samples, seed=7)
    first_split = next(name for name, group in split.items() if samples[0].image_path in {item.image_path for item in group})
    duplicate_split = next(name for name, group in split.items() if samples[1].image_path in {item.image_path for item in group})
    assert first_split == duplicate_split
    with pytest.raises(ValueError, match="identical image"):
        assert_no_leakage([samples[0]], [samples[1]])


def test_candidate_ranking_and_metrics():
    ranked = rank_candidates("AMOXICILIN", ["AMOXICILLIN", "ASPIRIN"], top_k=2)
    assert ranked[0]["candidate"] == "AMOXICILLIN"
    assert ranked[0]["score"] <= 1.0
    assert ranked[0]["score_type"] == "normalized_edit_similarity"
    rows = [
        {"reference": "AMOXICILLIN", "prediction": "AMOXICILLIN", "sample_type": "real"},
        {"reference": "ASPIRIN", "prediction": "ASPRIN", "sample_type": "synthetic"},
    ]
    metrics = evaluate_rows(rows, lexicon=["AMOXICILLIN", "ASPIRIN"])
    assert metrics["overall"]["exact_match_accuracy"] == 0.5
    assert metrics["overall"]["top_1_accuracy"] == 1.0
    assert set(metrics["by_sample_type"]) == {"real", "synthetic"}
    assert edit_distance("abc", "adc") == 1


def test_lexicon_loader_accepts_uploaded_file_bytes():
    lexicon = BytesIO(b"Aspirin\n aspirin \nParacetamol\n")
    assert load_lexicon(lexicon) == ["ASPIRIN", "PARACETAMOL"]


def test_ctc_decoder_collapses_repeats_but_not_repeats_separated_by_blank():
    torch = pytest.importorskip("torch")
    from src.prediction import greedy_ctc_decode

    # class 0 is blank; 1='a', 2='b'.
    ids = torch.tensor([[1], [1], [0], [1], [2], [2]])
    logits = torch.full((6, 1, 3), -10.0)
    logits.scatter_(2, ids.unsqueeze(-1), 0.0)
    assert greedy_ctc_decode(logits, "ab") == ["aab"]
