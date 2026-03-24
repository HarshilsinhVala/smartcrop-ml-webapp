from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import cv2
import uuid
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

CROP_MODEL_PATH    = os.path.join(MODEL_DIR, "crop_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CNN_MODEL_PATH     = os.path.join(MODEL_DIR, "cnn_model.pth")

# ─── Disease Class Names (38 classes — matches Train/ folder exactly) ─────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
logging.info(f"✅ Loaded {len(CLASS_NAMES)} class names")

# ─── CNN Model Architecture (matches retrained model exactly) ─────────────────
IMG_SIZE = 128

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        flat_size = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ─── Load Crop Recommendation Model ──────────────────────────────────────────
try:
    with open(CROP_MODEL_PATH, "rb") as f:
        crop_model = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    logging.info("✅ Crop Model and Label Encoder loaded successfully!")
except Exception as e:
    logging.error(f"🚨 Error loading crop model: {e}")
    crop_model, label_encoder = None, None

# ─── Load CNN Model (PyTorch) ─────────────────────────────────────────────────
cnn_model = None
try:
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

    # Auto-detect num_classes from last weight layer
    last_key    = [k for k in checkpoint.keys() if 'weight' in k][-1]
    num_classes = checkpoint[last_key].shape[0]
    logging.info(f"🔍 Detected num_classes = {num_classes}")

    if num_classes != len(CLASS_NAMES):
        logging.warning(
            f"⚠️  Model has {num_classes} classes but CLASS_NAMES has {len(CLASS_NAMES)}. "
            f"Predictions beyond index {len(CLASS_NAMES)-1} will show 'Unknown Class N'."
        )

    cnn_model = ImprovedCNN(num_classes)
    cnn_model.load_state_dict(checkpoint)
    cnn_model.eval()
    logging.info("✅ PyTorch CNN Model loaded successfully!")
except Exception as e:
    logging.error(f"🚨 Error loading CNN model: {e}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"❌ Image not found at {image_path}")
            return None
        # Convert BGR (OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        # Apply same normalization used during training
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        img  = np.transpose(img, (2, 0, 1))
        img  = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img
    except Exception as e:
        logging.error(f"🚨 Image preprocessing failed: {e}")
        return None

# ─── Crop Prediction Route ────────────────────────────────────────────────────
CROP_FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        if crop_model is None or label_encoder is None:
            return jsonify({"error": "Crop model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        missing = [k for k in CROP_FEATURE_NAMES if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = pd.DataFrame(
            [[data[k] for k in CROP_FEATURE_NAMES]],
            columns=CROP_FEATURE_NAMES
        )

        prediction = crop_model.predict(features)[0]
        crop_name  = label_encoder.inverse_transform([prediction])[0]

        confidence = None
        if hasattr(crop_model, "predict_proba"):
            proba      = crop_model.predict_proba(features)[0]
            confidence = round(float(proba.max()), 4)

        response = {"crop": crop_name}
        if confidence is not None:
            response["confidence"] = confidence

        return jsonify(response)

    except Exception as e:
        logging.error(f"🚨 Crop prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

# ─── Disease Detection Route ──────────────────────────────────────────────────
@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        if cnn_model is None:
            return jsonify({"error": "CNN model not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        image_path      = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(image_path)

        image = preprocess_image(image_path)
        if image is None:
            return jsonify({"error": "Failed to process image"}), 400

        with torch.no_grad():
            output      = cnn_model(image)
            probs       = torch.nn.functional.softmax(output, dim=1)[0]
            class_index = torch.argmax(probs).item()
            confidence  = round(probs[class_index].item(), 4)

        disease_name = (
            CLASS_NAMES[class_index]
            if class_index < len(CLASS_NAMES)
            else f"Unknown Class {class_index}"
        )

        # Clean display name: "Tomato___Late_blight" → "Tomato — Late Blight"
        display_name = disease_name.replace("___", " — ").replace("_", " ").title()

        try:
            os.remove(image_path)
        except Exception:
            pass

        return jsonify({
            "predicted_class": class_index,
            "disease":         display_name,
            "raw_class":       disease_name,
            "confidence":      confidence,
        })

    except Exception as e:
        logging.error(f"🚨 Disease prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ─── Health Check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "crop_model":   crop_model is not None,
        "cnn_model":    cnn_model is not None,
        "class_names":  len(CLASS_NAMES),
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)