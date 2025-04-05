# app.py

import sys
import asyncio
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch

# Fix for Windows event loop error
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Preprocessing function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morph operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

# OCR using Tesseract
def extract_text_tesseract(image):
    processed = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=custom_config)
    return text.strip()

# OCR using TrOCR
def extract_text_trocr(image_pil):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# Blur detection
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < 100

# Streamlit app
st.title("📝 Smart OCR: Handwritten Text Recognition")
uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_cv = np.array(image_pil)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    if is_blurry(image_cv):
        st.warning("⚠️ The image appears blurry. Try retaking it in better lighting.")

    # Run both OCR engines
    st.info("Running TrOCR model...")
    trocr_text = extract_text_trocr(image_pil)

    st.info("Running Tesseract OCR...")
    tesseract_text = extract_text_tesseract(image_cv)

    # Display both results
    st.subheader("TrOCR Output")
    st.write(trocr_text or "[No text detected]")

    st.subheader("Tesseract Output")
    st.write(tesseract_text or "[No text detected]")

    # Choose better output
    final_text = trocr_text if len(trocr_text) >= len(tesseract_text) else tesseract_text
    st.success("✅ Final Predicted Text:")
    st.code(final_text)

    st.caption("Tip: For best results, avoid shadows and blur. But this app handles many real-world issues too!")
