import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# -----------------------------
# Load image
# -----------------------------
def load_img(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# -----------------------------
# Convert tensor to image
# -----------------------------
def tensor_to_image(tensor):
    tensor = tensor.numpy()
    tensor = np.squeeze(tensor, axis=0)
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

# -----------------------------
# Load fast NST model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

# -----------------------------
# Fast Style Transfer
# -----------------------------
def run_style_transfer(content_img, style_img):
    model = load_model()
    stylized_image = model(content_img, style_img)[0]
    return stylized_image

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Style Transfer", layout="centered")

st.title("AI Style Transfer")
st.write("Real-time style transfer using deep learning (Transfer Learning)")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

alpha = st.slider("Style Intensity", 0.0, 1.0, 0.5)

if content_file and style_file:
    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    st.subheader("Content Image")
    st.image(content_img, width=250)

    st.subheader("Style Image")
    st.image(style_img, width=250)

    content_tensor = load_img(content_img)
    style_tensor = load_img(style_img)

    if st.button("Apply Style Transfer"):
        with st.spinner("Applying style..."):
            stylized = run_style_transfer(content_tensor, style_tensor)

            # Blend for intensity control (your feature)
            blended = alpha * stylized + (1 - alpha) * content_tensor

            result = tensor_to_image(blended)

        st.subheader("Result")
        st.image(result, width=300)