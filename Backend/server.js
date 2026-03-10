const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const authRoutes = require("./routes/authRoutes"); 
const auctionRoutes = require("./routes/auctionRoutes");
const dealsRoutes = require("./routes/dealsRoutes");

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// 🔹 MongoDB Connection
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB connected successfully"))
  .catch((err) => {
    console.error("❌ MongoDB connection error:", err);
    process.exit(1);
  });

// 🔹 Routes
app.use("/auth", authRoutes); 
app.use("/auctions", auctionRoutes);
app.use("/deals", dealsRoutes);

// 🔹 Root Route
app.get("/", (req, res) => {
  res.send("🚀 Server is running! Visit /auth, /auctions, or /deals");
});

// 🔹 Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
