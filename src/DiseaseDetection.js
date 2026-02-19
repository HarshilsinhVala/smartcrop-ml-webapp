import React, { useState } from "react";
import axios from "axios";
import "./DiseaseDetection.css";

const DiseaseDetection = () => {
  const [file, setFile] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle Image Selection & Preview
  const handleImageUpload = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setUploadedImage(URL.createObjectURL(selectedFile));
      setPrediction(null);
      setError(null);
    }
  };

  // Handle Image Upload & Prediction
  const handleUpload = async () => {
    if (!file) {
      setError("❌ Please select an image.");
      return;
    }

    setError(null);
    setPrediction(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    try {
      console.log("📤 Uploading image...");
      const response = await axios.post("http://127.0.0.1:5000/predict_disease", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("📩 Response received:", response.data);

      if (response.data.error) {
        setError(response.data.error);
      } else {
        setPrediction({
          disease: response.data.disease, // ✅ Display Disease Name
          confidence: response.data.confidence,
        });
      }
    } catch (error) {
      console.error("❌ Upload Error:", error);
      setError("⚠ Failed to connect to the server. Ensure Flask is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="disease-detection-container">
      <h1>🌿 Plant Disease Detection</h1>
      <p>Upload an image to detect plant diseases.</p>

      {/* File Input */}
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      {/* Image Preview */}
      {uploadedImage && <img src={uploadedImage} alt="Uploaded Preview" className="uploaded-image" />}

      {/* Upload Button */}
      <button onClick={handleUpload} className="upload-button" disabled={loading}>
        {loading ? "⏳ Analyzing..." : "📤 Upload & Predict"}
      </button>

      {/* Prediction Result */}
      {prediction && (
        <div className="result">
          <h3>🩺 Diagnosis:</h3>
          <p>
            🌱 <strong>Disease:</strong> {prediction.disease}
          </p>
          <p>
            🎯 <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}

      {/* Error Message */}
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default DiseaseDetection;
