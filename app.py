# app.py

import sys
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
import pytesseract
import numpy as np
import cv2

# Fix Windows async loop issue
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Title
st.title("📝 Smart Note Buddy - Handwritten Text Recognition")

# Image Preprocessing Function for Tesseract
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

# Tesseract OCR fallback
def pytesseract_ocr(uploaded_image):
    image = np.array(Image.open(uploaded_image).convert("RGB"))
    processed = preprocess_image(image)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=config)
    return text

# Upload image
uploaded_file = st.file_uploader("📤 Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Load TrOCR model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # TrOCR prediction
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("🔍 TrOCR Prediction:")
    st.write(predicted_text)

    # If TrOCR gives unexpected results, offer Tesseract fallback
    if st.checkbox("Fallback to Tesseract OCR if result is wrong"):
        text_tesseract = pytesseract_ocr(uploaded_file)
        st.subheader("🔁 Tesseract OCR Result:")
        st.write(text_tesseract)
