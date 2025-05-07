import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ------------------ Setup ------------------

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection with Grad-CAM")

img_size = 128  # Consistent with training

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_model.h5")

model = load_model()

# Ensure these class names match your training order
class_names = [
    "Pepper__bell___healthy",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Potato__Late_blight",
    "Potato__Early_blight",
    "Potato__healthy",
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Late_blight"
]

# ------------------ Upload ------------------

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

# ------------------ Prediction & Grad-CAM ------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((img_size, img_size))
    img_array = np.array(image_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    st.image(image, caption='🖼 Uploaded Image', use_container_width=True)

    # Prediction
    prediction = model.predict(input_tensor)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    st.success(f"🌱 **Predicted Class:** `{predicted_class}`")

    # ------------------ Grad-CAM ------------------

    st.subheader("🔍 Grad-CAM Visualization")

    # Use the last conv layer by name
    last_conv_layer_name = "Conv_1"  # Works for MobileNetV2 — change if needed

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        loss = predictions[:, predicted_class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Correct reduction axis for [batch, height, width, channels]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Create heatmap overlay
    heatmap = cv2.resize(heatmap.numpy(), (img_size, img_size))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(image.resize((img_size, img_size)))
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    st.image(overlay, caption="🔥 Model Focus (Grad-CAM)", use_container_width=True)
