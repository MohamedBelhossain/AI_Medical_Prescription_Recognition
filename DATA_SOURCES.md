# Data sources and design references

Status reviewed 15 November 2026. This project does not automatically download these datasets. Confirm the terms shown for the exact version you download, keep attribution/license notices, and do not commit patient data.

## Handwriting datasets

| Source | Contents and access | License / use status |
| --- | --- | --- |
| [Doctor's Handwritten Prescription BD on Kaggle](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset) and [Hugging Face Lance mirror](https://huggingface.co/datasets/lance-format/handwriting-ocr) | The mirror card reports 4,680 cropped medicine-name images, labels for medicine and generic names, and train/validation/test splits. Kaggle may require an account; the mirror is publicly viewable/downloadable. | The Hugging Face mirror metadata says ODbL 1.0. Keep its attribution and share-alike conditions with redistributed database material. The card notes repeated samples; deduplicate exact image copies before splitting. The Kaggle page could not be inspected automatically, so use the mirror's stated license only for that mirror/version. |
| [RxHandBD on Mendeley Data, version 3](https://data.mendeley.com/datasets/dsb5r6vskg/3) | The record describes 5,578 cropped word images, 1,559 unique labels, and CSV train/test labels. | Mendeley version 3 displays CC BY 4.0. Attribute the authors and record version. The separately supplied [Zenodo record](https://zenodo.org/records/18478741) has a blank Rights/License field in the page metadata checked here; do not assume the Zenodo files have Mendeley's license. Prefer the Mendeley v3 record unless the Zenodo license is clarified. |
| [HP_DocPres on IEEE DataPort](https://ieee-dataport.org/documents/dataset-classification-handwritten-and-printed-text-doctors-prescription) | Handwritten/printed/mixed prescription-text classification, not a medicine word-transcription set. The page could not be inspected by the automated browser; it may require an IEEE account. | License not verified. Download manually if access is available and check the record's terms before use. |
| [MIRAGE 100-record sample on Hugging Face](https://huggingface.co/datasets/chaithanyakota/100-handwritten-medical-records) | 100 page-level records with medicine annotations. The paper discusses a much larger simulated corpus, but the linked public sample contains 100 rows. The dataset card's README is empty. | The Hugging Face page lists CC BY-ND 4.0. Do not augment, transform, or redistribute derivatives under this license without permission. Review the image content and privacy before use; this project does not ingest it automatically. |
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) | General English handwriting forms, lines, sentences, and words; registration is requested. Useful only for general pretraining. | The official page says non-commercial research use only. Register and cite the IAM paper if used; do not use it in a commercial deployment without authorization. |
| [Existing 46-image sample](https://huggingface.co/datasets/avi-kai/Medical_Prescription_Handwritten_Words) | Single-word demo images already exist locally under `dataset/raw/images/`; the repository's ignored local copy can be recreated with `python scripts/download_dataset.py`. | Hugging Face lists MIT. This small set is only a smoke-test sample, not a representative benchmark. |

## Text lexicons and normalization data

- [RxNorm licensing and access](https://www.nlm.nih.gov/research/umls/rxnorm/docs/termsofservice.html): the full RxNorm release requires a free UMLS license/account, and included third-party terminology sources can have additional restrictions. NLM's Current Prescribable Content release is separately listed as requiring no license; check the exact release and terms before downloading. RxNorm is US-centric.
- [openFDA terms](https://open.fda.gov/terms/): most openFDA content is public domain/CC0, but the terms call out exceptions for third-party content. Check the selected endpoint/data fields rather than assuming every returned field has identical rights.
- [DrugBank academic research access](https://go.drugbank.com/academic_research/) and [terms](https://trust.drugbank.com/drugbank-trust-center/terms-of-use): DrugBank requires an applicable license. Academic access is limited to qualifying academic, non-clinical use; do not copy its names into this project unless the license for the intended use allows it.
- [IndicRxNorm-Gemma3-270M-LoRA](https://huggingface.co/AXONVERTEX-AI-RESEARCH/IndicRxNorm-Gemma3-270M-LoRA): the referenced Hugging Face location could not be inspected in this review. It appears to be a model/adaptor rather than a plain drug-name lexicon; confirm the repository and model/data licenses before use. It is not used here.

The lexicon interface accepts a user-reviewed text file with one candidate name per line. No medication names are shipped as an authoritative vocabulary.

## Reference papers and pipeline choices

- [MIRAGE (2024)](https://arxiv.org/abs/2410.09729): evaluates multimodal models on a large simulated prescription corpus, reports medicine-name/dosage extraction separately from all-field extraction, and explores spaced-character targets. The paper's scores and compute setup are specific to its simulated data and are not targets or expected results for this project.
- [Weakly supervised extraction from handwritten images (2023)](https://arxiv.org/abs/2306.06823): motivates localized medicine-line recognition and a domain vocabulary/language model trained with synthetic name lines. This project implements the narrower crop-recognition and lexicon-ranking pieces first; page localization is deferred.
- [Multi-head attention and TrOCR medicine extraction (2024)](https://arxiv.org/abs/2412.18199): describes segmentation plus TrOCR and augmentation. Its reported values use a separate dataset and should not be compared directly to results here.
- [Benchmark dataset for OCR of drug names (2025)](https://sebhau.edu.ly/journal/sucp/article/view/4228): relevant dataset-design reference; no downloadable, licensed data source was verified from the linked paper page.

## Local data rules

Raw and processed data under `data/` are git-ignored. Keep the source name, license, version, download date, and any writer/document identifiers in local metadata. Do not merge real patient images into a public repository. Synthetic samples must stay labeled `sample_type=synthetic` and be reported separately from real images.
