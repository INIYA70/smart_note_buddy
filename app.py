import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

st.set_page_config(page_title="Smart Note Buddy", layout="wide")
st.title("📝 Smart Note Buddy")
st.markdown("Extract handwritten text from your notes with AI.")

uploaded_file = st.file_uploader("Upload an image of your notes", type=["jpg", "jpeg", "png"])

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def predict_text(image: Image.Image) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        angle = st.slider("Rotate Image", -180, 180, 0)
        if angle != 0:
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            rotated = rotate_image(open_cv_image, angle)
            image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
            st.image(image, caption="Rotated Image", use_column_width=True)

    with col2:
        if st.button("Extract Text"):
            with st.spinner("Analyzing handwriting..."):
                extracted_text = predict_text(image)
                st.success("Text Extraction Complete ✅")
                st.text_area("Extracted Text", extracted_text, height=200)
