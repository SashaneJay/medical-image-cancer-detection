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
This application uses a MobileNetV2 deep learning model trained on
277,524 histopathology image patches.

### Classes
- Non-Cancer
- Cancer

For educational and research purposes only.
""")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    image_resized = image.resize(IMG_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)[0][0]

    st.subheader("Prediction Result")

    if prediction >= 0.5:
        st.error(f"Cancer detected probability: {prediction:.2%}")
    else:
        st.success(f"Non-cancer probability: {(1 - prediction):.2%}")

    st.write("Raw cancer probability:", round(float(prediction), 4))