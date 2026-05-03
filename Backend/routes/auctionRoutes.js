const express = require("express");
const Auction = require("../models/Auction");
const authMiddleware = require("../middleware/authMiddleware");

const router = express.Router();

function requireAdmin(req, res, next) {
  if (!req.user || req.user.role !== "admin") {
    return res.status(403).json({ error: "Admin access required" });
  }
  next();
}

// Get all auctions
router.get("/", async (req, res) => {
  try {
    const auctions = await Auction.find();
    res.json(auctions);
  } catch (err) {
    res.status(500).json({ error: "Error fetching auctions" });
  }
});

// Add a new auction
router.post("/", authMiddleware, requireAdmin, async (req, res) => {
  try {
    const { name, description, quantity, price } = req.body;
    const newAuction = new Auction({ name, description, quantity, price });
    await newAuction.save();
    res.json({ message: "Auction created successfully", auction: newAuction });
  } catch (err) {
    res.status(500).json({ error: "Error adding auction" });
  }
});

// ✅ Update auction (quantity & price) when making a deal
router.put("/:id", authMiddleware, requireAdmin, async (req, res) => {
  try {
    const { quantity, price } = req.body;
    const updatedAuction = await Auction.findByIdAndUpdate(
      req.params.id,
      { quantity, price },
      { new: true }
    );

    if (!updatedAuction) {
      return res.status(404).json({ error: "Auction not found" });
    }

    res.json({ message: "Auction updated successfully", auction: updatedAuction });
  } catch (err) {
    res.status(500).json({ error: "Error updating auction" });
  }
});

module.exports = router;