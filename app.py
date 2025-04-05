# app.py

import sys

# Fix for "no running event loop" error on Windows
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# -----------------------------
# Tesseract Preprocessing Function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

# -----------------------------
# Tesseract OCR Function
def extract_text_tesseract(uploaded_image):
    image = np.array(Image.open(uploaded_image))
    processed = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=custom_config)
    return text

# -----------------------------
# Streamlit App
st.title("📝 Smart Note Buddy - Handwritten Text Recognition")

uploaded_file = st.file_uploader("Upload an image of handwritten or printed text", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- TrOCR Section ---
    st.subheader("🔍 Predicted Text using TrOCR")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        st.success(predicted_text if predicted_text.strip() else "No text found.")
    except Exception as e:
        st.error(f"TrOCR failed: {e}")

    # --- Tesseract Section ---
    st.subheader("🧠 Backup Text Extraction using Tesseract")
    try:
        tesseract_text = extract_text_tesseract(uploaded_file)
        st.success(tesseract_text if tesseract_text.strip() else "No text found.")
    except Exception as e:
        st.error(f"Tesseract failed: {e}")
