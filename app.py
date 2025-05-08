import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection with Grad-CAM")

img_size = 128

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_model.h5")

model = load_model()

# Confirm these match the training set's class_indices
class_names = sorted([
    "Pepper__bell___healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato__healthy",
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy"
])

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((img_size, img_size))
    img_array = np.array(image_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    prediction = model.predict(input_tensor)
    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]

    st.success(f"🌱 **Predicted Class:** `{predicted_class}`")
