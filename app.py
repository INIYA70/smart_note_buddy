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
        M = cv2.getRotationMatrix2D((w // 2, h
