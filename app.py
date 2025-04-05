# app.py

import sys
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch

# ---------------- Preprocessing Function ---------------- #
def preprocess_image(image):
    img = np.array(image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise with Gaussian Blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = thresh.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = thresh

    # Convert back to PIL for model
    return Image.fromarray(deskewed)

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Smart Note Buddy 📝", layout="centered")
st.title("🧠 Smart Note Buddy - Handwritten OCR")

uploaded_file = st.file_uploader("📷 Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Original Image", use_container_width=True)

    with st.spinner("⏳ Preprocessing and recognizing text..."):
        preprocessed_image = preprocess_image(image)
        st.image(preprocessed_image, caption="🧪 Preprocessed Image", use_container_width=True)

        # Load large model (swap with 'base' if needed)
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

        # Predict
        pixel_values = processor(images=preprocessed_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("📄 Extracted Text:")
    st.success(predicted_text)

    st.markdown("---")
    st.caption("🔧 Powered by TrOCR + Preprocessing | Created by Iniya Swedha 😊")

