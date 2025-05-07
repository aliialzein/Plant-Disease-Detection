# 🌿 Plant Disease Detection with Grad-CAM

This project detects plant diseases from leaf images using a fine-tuned MobileNetV2 deep learning model. It features an interactive web app built with Streamlit and provides Grad-CAM visualizations to highlight which parts of the leaf influenced the model's decision.

---

## 📸 Demo

Upload a photo of a leaf and get:
- The predicted disease class
- A heatmap showing the model's focus

---

## 🚀 Features

- ✅ Image classification using MobileNetV2
- ✅ Real-time predictions in a web app
- ✅ Grad-CAM visualization for model interpretability
- ✅ Lightweight and efficient for deployment

---

## 🧠 Model Info

- Trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Data augmented with rotation, zoom, flip, shift
- Fine-tuned on top 30 layers of MobileNetV2
- Saved as `best_model.h5`

---

## 📂 Project Structure

```plaintext
.
├── app.py               # Streamlit app
├── model/
│   └── best_model.h5    # Trained model
├── requirements.txt     # Python dependencies
└── README.md            # This file
