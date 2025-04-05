import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Smart Note Buddy", layout="wide")
st.title("📝 Smart Note Buddy")
st.subheader("Convert handwritten notes into editable text")

uploaded_file = st.file_uploader("Upload an image of handwritten notes", type=["png", "jpg", "jpeg"])

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption='Original Image', use_column_width=True)

    angle = st.slider("Rotate image (if text is tilted)", -180, 180, 0)
    rotated_image = rotate_image(image, angle)
    
    st.image(rotated_image, caption='Rotated Image', use_column_width=True)

    preprocessed = preprocess_image(rotated_image)

    st.image(preprocessed, caption='Preprocessed Image', use_column_width=True)

    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            text = extract_text(preprocessed)
        st.success("Here is the extracted text:")
        st.text_area("Output Text", text, height=300)
