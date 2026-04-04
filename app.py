import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Load image
# -----------------------------
def load_img(img):
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# -----------------------------
# Display image
# -----------------------------
def tensor_to_image(tensor):
    tensor = tensor.numpy().astype("uint8")[0]
    return Image.fromarray(tensor)

# -----------------------------
# VGG Model
# -----------------------------
@st.cache_resource
def load_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = tf.keras.Model([vgg.input], outputs)

    return model, style_layers, content_layers

# -----------------------------
# Gram matrix
# -----------------------------
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    shape = tf.shape(input_tensor)
    return result / tf.cast(shape[1]*shape[2], tf.float32)

# -----------------------------
# Feature extraction
# -----------------------------
def get_features(model, image, style_layers, content_layers):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    outputs = model(image)

    style_outputs = outputs[:len(style_layers)]
    content_outputs = outputs[len(style_layers):]

    style_features = [gram_matrix(x) for x in style_outputs]

    return style_features, content_outputs

# -----------------------------
# Style Transfer
# -----------------------------
def run_style_transfer(content_img, style_img):
    model, style_layers, content_layers = load_model()

    style_targets, _ = get_features(model, style_img, style_layers, content_layers)
    _, content_targets = get_features(model, content_img, style_layers, content_layers)

    generated = tf.Variable(content_img)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    for i in range(50):
        with tf.GradientTape() as tape:
            style_outputs, content_outputs = get_features(model, generated, style_layers, content_layers)

            style_loss = tf.add_n([
                tf.reduce_mean((style_outputs[j] - style_targets[j])**2)
                for j in range(len(style_layers))
            ])

            content_loss = tf.add_n([
                tf.reduce_mean((content_outputs[j] - content_targets[j])**2)
                for j in range(len(content_layers))
            ])

            loss = 1e-1 * style_loss + 1e3 * content_loss

        grad = tape.gradient(loss, generated)
        optimizer.apply_gradients([(grad, generated)])
        generated.assign(tf.clip_by_value(generated, 0.0, 255.0))

    return generated

# -----------------------------
# UI
# -----------------------------
st.title("🎨 AI Style Transfer App")
st.write("Upload images and control style intensity")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

alpha = st.slider("Style Intensity", 0.0, 1.0, 0.5)

if content_file and style_file:
    content_img = Image.open(content_file)
    style_img = Image.open(style_file)

    st.subheader("Original Content")
    st.image(content_img, width=250)

    st.subheader("Style Image")
    st.image(style_img, width=250)

    content_tensor = load_img(content_img)
    style_tensor = load_img(style_img)

    if st.button("Apply Style Transfer"):
        with st.spinner("Processing..."):
            stylized = run_style_transfer(content_tensor, style_tensor)

            # Blend (your feature)
            blended = alpha * stylized + (1 - alpha) * content_tensor

            result = tensor_to_image(blended)

        st.subheader("Result")
        st.image(result, width=300)