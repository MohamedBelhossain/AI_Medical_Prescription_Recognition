"""Streamlit interface for handwritten prescription word OCR."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import discover_images, load_manifest
from src.postprocessing import load_lexicon, rank_candidates
from src.prediction import predict_image


st.set_page_config(page_title="Prescription Word OCR", page_icon="💊", layout="centered")
st.title("Handwritten prescription word recognition")
st.write("Upload a clear crop containing one handwritten word or line. The first run downloads the TrOCR model.")
st.warning("Assistive research demo only. OCR and medicine matches can be wrong; never use this output as medically authoritative. A pharmacist or doctor must verify it.")
st.info("The uploaded image and confirmed text are not saved by this app. Avoid uploading identifiable patient information to shared deployments.")

uploaded = st.file_uploader("Prescription word image", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"])
model_name = st.text_input("Hugging Face model", value="microsoft/trocr-base-handwritten")
lexicon_file = st.file_uploader("Optional reviewed medicine vocabulary (.txt, one name per line)", type=["txt"])
if uploaded is not None and st.button("Recognize", type="primary"):
    with st.spinner("Recognizing handwriting…"):
        try:
            st.session_state["recognized_text"] = predict_image(uploaded, model_name=model_name.strip())
            st.session_state.pop("confirmed_text", None)
            st.session_state.pop("edited_text", None)
        except Exception as exc:
            st.error(f"Recognition failed: {exc}")

if "recognized_text" in st.session_state:
    raw_text = st.session_state["recognized_text"]
    st.subheader("OCR suggestion")
    st.code(raw_text or "No text recognized")
    initial_text = raw_text
    if lexicon_file is not None and raw_text:
        try:
            lexicon = load_lexicon(lexicon_file)
            candidates = rank_candidates(raw_text, lexicon, top_k=5)
            if candidates:
                st.caption("Vocabulary matches are edit-similarity scores, not probabilities or medical confidence.")
                st.dataframe(candidates, hide_index=True, use_container_width=True)
                initial_text = str(candidates[0]["candidate"])
        except Exception as exc:
            st.error(f"Could not read the vocabulary: {exc}")
    edited_text = st.text_input("Edit the medicine name or line, then confirm", value=initial_text, key="edited_text")
    if st.button("Confirm this text", type="primary", disabled=not edited_text.strip()):
        st.session_state["confirmed_text"] = edited_text.strip()
    if st.session_state.get("confirmed_text") == edited_text.strip() and edited_text.strip():
        st.success(f"Human-confirmed text: {st.session_state['confirmed_text']}")
    elif "confirmed_text" in st.session_state:
        st.session_state.pop("confirmed_text", None)

with st.expander("Explore the bundled word dataset"):
    manifest = ROOT / "dataset" / "labels" / "manifest.csv"
    samples = load_manifest(manifest) if manifest.exists() else discover_images(ROOT / "dataset" / "raw")
    st.write(f"{len(samples)} labeled word images available.")
    if samples:
        st.dataframe([{"image": sample.image_path.name, "label": sample.text} for sample in samples], use_container_width=True)
