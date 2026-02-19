import React, { useState } from "react";
import axios from "axios";
import "./CropPrediction.css";

const CropPrediction = () => {
  const [formData, setFormData] = useState({
    N: "",
    P: "",
    K: "",
    temperature: "",
    humidity: "",
    ph: "",
    rainfall: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    setLoading(true);

    try {
      const formattedData = {
        N: parseFloat(formData.N) || 0,
        P: parseFloat(formData.P) || 0,
        K: parseFloat(formData.K) || 0,
        temperature: parseFloat(formData.temperature) || 0,
        humidity: parseFloat(formData.humidity) || 0,
        ph: parseFloat(formData.ph) || 0,
        rainfall: parseFloat(formData.rainfall) || 0,
      };

      console.log("📤 Sending data:", formattedData);

      const response = await axios.post("http://127.0.0.1:5000/predict_crop", formattedData, {
        headers: { "Content-Type": "application/json" },
      });

      console.log("📩 Response received:", response.data);

      if (response.data.error) {
        setError(response.data.error);
      } else {
        setPrediction(response.data.crop);
      }
    } catch (err) {
      console.error("❌ Prediction Error:", err);
      setError("⚠ Failed to connect to the server. Ensure Flask is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="crop-prediction-page">
      <h1 className="title">🌱 SmartCrop: Intelligent Crop Recommendation</h1>
      <p className="subtitle">Find out the most suitable crop to grow on your farm 👨‍🌾</p>

      <form className="form" onSubmit={handleSubmit}>
        {Object.keys(formData).map((key) => (
          <div className="form-group" key={key}>
            <label htmlFor={key}>{key.toUpperCase()}:</label>
            <input type="number" id={key} name={key} step="0.1" value={formData[key]} onChange={handleChange} required />
          </div>
        ))}

        <button type="submit" className="predict-button" disabled={loading}>
          {loading ? "⏳ Predicting..." : "🔍 Predict"}
        </button>
      </form>

      {prediction && <h3 className="result">🌾 Recommended Crop: <strong>{prediction}</strong></h3>}
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default CropPrediction;
