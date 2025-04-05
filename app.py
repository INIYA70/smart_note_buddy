# app.py

import sys
import numpy as np
import cv2
import torch
from PIL import Image
import streamlit as st
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Fix for Windows platform issue with asyncio
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Title
st.title("📝 Handwritten Text Recognition using TrOCR (Enhanced Accuracy)")

# Upload section
uploaded_file = st.file_uploader("Upload an image of handwritten text", type=["png", "jpg", "jpeg"])

# Preprocessing function for TrOCR
def preprocess_for_trocr(pil_image):
    # Convert PIL to OpenCV format
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to standard input size for TrOCR (you can tweak this)
    resized = cv2.resize(gray, (384, 384))
    
    # Normalize image (if needed, we can do CLAHE or similar here too)
    norm = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Return preprocessed PIL image
    return Image.fromarray(norm)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing image..."):
        # Preprocess image
        processed_image = preprocess_for_trocr(image)

        # Load larger model for better accuracy
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

        # Run inference
        pixel_values = processor(images=processed_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Display result
    st.subheader("🔍 Predicted Text:")
    st.write(predicted_text)
