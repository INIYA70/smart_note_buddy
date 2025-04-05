# app.py

import sys

# Fix for "no running event loop" error
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch

# Title
st.title("📝 Handwritten Text Recognition using TrOCR")

# File uploader
uploaded_file = st.file_uploader("Upload an image of handwritten text", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load pretrained processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Inference
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Display result
    st.subheader("🔍 Predicted Text:")
    st.write(predicted_text)
