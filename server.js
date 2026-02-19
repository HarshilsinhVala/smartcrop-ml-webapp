const express = require("express");
const multer = require("multer");
const cors = require("cors");
const axios = require("axios");
const path = require("path");
const bodyParser = require("body-parser");
const { PythonShell } = require("python-shell");

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Configure multer for image uploads
const storage = multer.diskStorage({
  destination: "./uploads/",
  filename: (req, file, cb) => {
    cb(null, file.fieldname + "-" + Date.now() + path.extname(file.originalname));
  },
});
const upload = multer({ storage });

// Handle image upload and send to Python backend
app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    const imagePath = req.file.path;
    const response = await axios.post("http://127.0.0.1:5000/predict", { imagePath });
    res.json(response.data);
  } catch (error) {
    console.error("Image processing error:", error);
    res.status(500).json({ error: "Failed to process image" });
  }
});

// Handle crop prediction request
app.post("/predict", async (req, res) => {
  try {
    const { nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall } = req.body;

    let options = {
      mode: "text",
      pythonOptions: ["-u"],
      scriptPath: "./",
      args: [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
    };

    PythonShell.run("predict.py", options, (err, results) => {
      if (err) {
        console.error("Python error:", err);
        return res.status(500).json({ error: "Prediction failed" });
      }
      res.json({ crop: results[0] });
    });
  } catch (error) {
    console.error("Server error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Start the server
const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
