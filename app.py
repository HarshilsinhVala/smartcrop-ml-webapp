from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import torch
import cv2
import uuid  # Generate unique filenames
import logging
from torchvision import datasets, transforms  # Import for class names

# Fix import for CNN model
try:
    from model import CNN  # Ensure model.py exists in your project folder
except ImportError:
    logging.error("❌ model.py not found. Ensure it exists in the same directory.")
    CNN = None  # Prevents crashes if model.py is missing

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all routes for frontend access

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Path to models directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")  # Folder to store uploaded images
TRAIN_DIR = os.path.join(BASE_DIR, "Train")  # Training folder (to get class names)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure 'uploads' folder exists

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Restrict file types for security

CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")  # PyTorch model

# ✅ Load class names dynamically from dataset
try:
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    CLASS_NAMES = train_dataset.classes  # Extract class names from dataset folders
    logging.info(f"🔍 Loaded {len(CLASS_NAMES)} class names: {CLASS_NAMES}")
except Exception as e:
    logging.error(f"🚨 Error loading class names: {e}")
    CLASS_NAMES = []

# Load Crop Recommendation Model
try:
    with open(CROP_MODEL_PATH, "rb") as crop_model_file:
        model = pickle.load(crop_model_file)
    with open(LABEL_ENCODER_PATH, "rb") as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    logging.info("✅ Crop Model and Label Encoder loaded successfully!")
except Exception as e:
    logging.error(f"🚨 Error loading crop model: {e}")
    model, label_encoder = None, None  

# Load CNN Model (PyTorch)
try:
    if CNN is not None:
        # Detect num_classes dynamically from the saved model
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu'))
        num_classes = checkpoint['fc2.weight'].shape[0]  # Extract num_classes
        logging.info(f"🔍 Detected num_classes = {num_classes}")

        cnn_model = CNN(num_classes)  # Use the correct num_classes
        cnn_model.load_state_dict(checkpoint)
        cnn_model.eval()  # Set to evaluation mode
        logging.info("✅ PyTorch CNN Model loaded successfully!")
    else:
        cnn_model = None
except Exception as e:
    logging.error(f"🚨 Error loading CNN model: {e}")
    cnn_model = None

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing for CNN Model
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"❌ Error: Image not found at {image_path}")
            return None
        img = cv2.resize(img, (128, 128))  # Resize to 128x128
        img = img.astype('float32') / 255.0  # Normalize pixel values
        img = np.transpose(img, (2, 0, 1))  # Convert to PyTorch format (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return img
    except Exception as e:
        logging.error(f"🚨 Image preprocessing failed: {e}")
        return None

# ✅ Updated API Route for Plant Disease Detection (Returns Disease Name)
@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        if cnn_model is None:
            return jsonify({"error": "CNN model not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

        # Save image
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(image_path)

        # Process image
        image = preprocess_image(image_path)
        if image is None:
            return jsonify({"error": "Failed to process image"}), 400

        # Predict using CNN model
        with torch.no_grad():
            predictions = cnn_model(image)
            class_index = torch.argmax(predictions, dim=1).item()
            confidence = torch.nn.functional.softmax(predictions, dim=1)[0][class_index].item()

        # ✅ Get disease name based on class_index
        disease_name = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else f"Unknown Class {class_index}"

        return jsonify({
            "predicted_class": class_index,
            "disease": disease_name,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        logging.error(f"🚨 Prediction failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# API Route for Crop Prediction
@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        if model is None or label_encoder is None:
            return jsonify({"error": "Crop model not loaded"}), 500

        data = request.get_json()  # Get JSON request data
        required_keys = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}
        
        # Check for missing fields
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Convert input data to numpy array
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'],
                              data['humidity'], data['ph'], data['rainfall']]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        crop_name = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({"crop": crop_name})

    except Exception as e:
        logging.error(f"🚨 Crop prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Ensures Flask runs on port 5000


