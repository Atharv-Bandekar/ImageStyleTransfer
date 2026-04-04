import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Load and preprocess image
# -----------------------------
def load_img(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, dtype=tf.float32)

def show_image(img):
    img = img.numpy().astype("uint8")[0]
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# -----------------------------
# Load images
# -----------------------------
content_image = load_img("content.jpg")
style_image = load_img("style.jpg")

print("Showing Content Image")
show_image(content_image)

print("Showing Style Image")
show_image(style_image)

# -----------------------------
# Load VGG19 (Transfer Learning)
# -----------------------------
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# -----------------------------
# Define layers
# -----------------------------
content_layers = ['block5_conv2']

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

# -----------------------------
# Create model
# -----------------------------
outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
model = tf.keras.Model([vgg.input], outputs)

# -----------------------------
# Gram Matrix
# -----------------------------
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    shape = tf.shape(input_tensor)
    return result / tf.cast(shape[1]*shape[2], tf.float32)

# -----------------------------
# Extract features
# -----------------------------
def get_features(image):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    outputs = model(image)

    style_outputs = outputs[:len(style_layers)]
    content_outputs = outputs[len(style_layers):]

    style_features = [gram_matrix(x) for x in style_outputs]

    return style_features, content_outputs

style_targets, _ = get_features(style_image)
_, content_targets = get_features(content_image)

# -----------------------------
# Generated image
# -----------------------------
generated_image = tf.Variable(content_image)

# -----------------------------
# Loss function (FIXED)
# -----------------------------
def compute_loss(gen_image):
    style_outputs, content_outputs = get_features(gen_image)

    style_loss = tf.add_n([
        tf.reduce_mean((style_outputs[i] - style_targets[i])**2)
        for i in range(len(style_layers))
    ])

    content_loss = tf.add_n([
        tf.reduce_mean((content_outputs[i] - content_targets[i])**2)
        for i in range(len(content_layers))
    ])

    # Balanced weights
    total_loss = 1e-1 * style_loss + 1e3 * content_loss
    return total_loss

# -----------------------------
# Optimizer (FIXED)
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# -----------------------------
# Training step
# -----------------------------
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        loss = compute_loss(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 255.0))

# -----------------------------
# Training loop (INCREASED)
# -----------------------------
print("Training started...")

for i in range(300):
    train_step(generated_image)

    # if i % 50 == 0:
    #     print("Step:", i)
    #     show_image(generated_image)

# -----------------------------
# Final output
# -----------------------------
print("Final Stylized Image:")
show_image(generated_image)

# -----------------------------
# Intensity Control (YOUR FEATURE)
# -----------------------------
def blend_images(content, stylized, alpha):
    return alpha * stylized + (1 - alpha) * content

print("Showing intensity variations...")

for alpha in [0.0, 0.5, 1.0]:
    print(f"Alpha = {alpha}")
    blended = blend_images(content_image, generated_image, alpha)
    show_image(blended)