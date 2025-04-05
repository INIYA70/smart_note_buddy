import asyncio
import sys

# Fix for Python 3.10+ asyncio issues in Streamlit
if sys.platform.startswith('win') or sys.version_info >= (3, 10):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass

import streamlit as st
from PIL import Image
import io
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import torchvision.transforms as transforms

# Load processor and model from HuggingFace
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

processor, model = load_model()

st.title("📝 Smart Note Buddy")
uploaded_file = st.file_uploader("Upload a handwritten note", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Note", use_column_width=True)

    # Preprocess image
    transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
    pixel_values = transform(image).unsqueeze(0)

    st.info("🔍 Extracting handwritten text...")
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("🧾 Extracted Text")
    st.write(extracted_text)

    # Summarizer
    from transformers import pipeline
    summarizer = pipeline("summarization")
    summary = summarizer(extracted_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    st.subheader("🧠 Summary")
    st.write(summary)

    # QA System
    st.subheader("❓ Ask a Question")
    user_question = st.text_input("Type your question here:")
    if user_question:
        qa = pipeline("question-answering")
        answer = qa(question=user_question, context=extracted_text)
        st.write("✅ Answer:", answer['answer'])
