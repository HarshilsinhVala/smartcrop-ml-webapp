const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect("mongodb://localhost:27017/cropDB", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const cropSchema = new mongoose.Schema({
  Nitrogen: Number,
  Phosphorus: Number,
  Potassium: Number,
  Temperature: Number,
  Humidity: Number,
  pH: Number,
  Rainfall: Number,
  label: String, // Replace this with the actual column name for the crop type
});

const Crop = mongoose.model("Crop", cropSchema);

// API Endpoint for Prediction
app.post("/predict", async (req, res) => {
  try {
    const { Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall } = req.body;

    // Query MongoDB to find similar data
    const crop = await Crop.findOne({
      Nitrogen: { $lte: Nitrogen + 5, $gte: Nitrogen - 5 },
      Phosphorus: { $lte: Phosphorus + 5, $gte: Phosphorus - 5 },
      Potassium: { $lte: Potassium + 5, $gte: Potassium - 5 },
      Temperature: { $lte: Temperature + 2, $gte: Temperature - 2 },
      Humidity: { $lte: Humidity + 5, $gte: Humidity - 5 },
      pH: { $lte: pH + 0.5, $gte: pH - 0.5 },
      Rainfall: { $lte: Rainfall + 10, $gte: Rainfall - 10 },
    });

    if (crop) {
      res.json({ success: true, crop: crop.label });
    } else {
      res.json({ success: false, message: "No matching crop found." });
    }
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Start the Server
const PORT = 3000;
app.listen(PORT, () => {
  app.listen(PORT, () => console.log(`Server running on port http://localhost:${3000}/`));
});