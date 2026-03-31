import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ==============================
# Load Model and Labels
# ==============================

MODEL_PATH = "retrained_graph.keras"
LABELS_PATH = "retrained_labels.txt"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file retrained_graph.keras not found.")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("retrained_labels.txt not found.")

# Load trained Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]


# ==============================
# Prediction Function
# ==============================

def predict_image(img_path):
    """
    Takes image path and returns predicted label
    """

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(predictions[0])) * 100

    return {
        "label": predicted_label,
        "confidence": round(confidence, 2)
    }
