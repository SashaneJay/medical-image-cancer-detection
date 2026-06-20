import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_cancer_classifier.keras"

IMG_SIZE = (50, 50)

st.title("Breast Cancer Histopathology Classifier")

st.markdown("""
This application uses a MobileNetV2 deep learning model to classify breast
histopathology image patches as **Cancer** or **Non-Cancer**.

**Dataset:** 277,524 histopathology image patches  
**Purpose:** Educational machine learning project only.
""")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a histopathology image patch",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    image_resized = image.resize(IMG_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    cancer_probability = float(model.predict(image_array)[0][0])
    non_cancer_probability = 1 - cancer_probability

    st.subheader("Prediction Result")

    if cancer_probability >= 0.5:
        st.error(f"Cancer detected probability: {cancer_probability:.2%}")
    else:
        st.success(f"Non-cancer probability: {non_cancer_probability:.2%}")

    st.write("Raw cancer probability:", round(cancer_probability, 4))

    st.warning(
        "Disclaimer: This project is for educational purposes only and is not a medical diagnostic tool."
    )