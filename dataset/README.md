# Bundled sample dataset

The `raw/images` files were downloaded from [avi-kai/Medical_Prescription_Handwritten_Words](https://huggingface.co/datasets/avi-kai/Medical_Prescription_Handwritten_Words) on Hugging Face. The dataset page identifies the license as MIT and describes 46 image samples. Each filename is used as its transcription; the generated manifest is at `labels/manifest.csv`.

This is a tiny demonstration dataset of cropped individual words. It is not representative enough for model training or clinical use. Run `python scripts/download_dataset.py` to retrieve it again.
