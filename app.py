# app.py

import sys
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Fix for "no running event loop" error (for Windows)
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Preprocess image for Tesseract
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

# Use Tesseract OCR as fallback
def tesseract_ocr(uploaded_image):
    image = np.array(Image.open(uploaded_image))
    processed = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=custom_config)
    return text.strip()

# Streamlit UI
st.title("📝 Handwritten Text Recognition using TrOCR + Tesseract Fallback")

uploaded_file = st.file_uploader("Upload an image of handwritten text", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load TrOCR model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Try TrOCR first
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        predicted_text = ""

    # If TrOCR fails or gives empty result, use Tesseract
    if not predicted_text or predicted_text.strip() == "":
        st.info("TrOCR result not confident. Switching to Tesseract OCR...")
        predicted_text = tesseract_ocr(uploaded_file)

    # Display result
    st.subheader("🔍 Predicted Text:")
    st.write(predicted_text if predicted_text else "Could not extract text.")
