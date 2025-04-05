import streamlit as st
from PIL import Image
import pytesseract
from transformers import pipeline

# Title
st.title("📝 Smart Note Buddy")

# File uploader
uploaded_file = st.file_uploader("Upload your handwritten note (image)", type=["png", "jpg", "jpeg"])

# Load OCR and Transformers
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Note", use_column_width=True)

    with st.spinner("🔍 Extracting text from image..."):
        extracted_text = pytesseract.image_to_string(image)
        st.subheader("🧾 Extracted Text")
        st.write(extracted_text)

    # Summarize text
    summarizer = pipeline("summarization")
    with st.spinner("🧠 Summarizing..."):
        summary = summarizer(extracted_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("🧠 Summary")
        st.write(summary)

    # QnA input
    st.subheader("❓ Ask a Question")
    user_question = st.text_input("Type your question here:")
    if user_question:
        qa = pipeline("question-answering")
        answer = qa(question=user_question, context=extracted_text)
        st.write("✅ Answer:", answer['answer'])
