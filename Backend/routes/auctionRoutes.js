const express = require("express");
const Auction = require("../models/Auction");

const router = express.Router();

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
router.post("/", async (req, res) => {
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
router.put("/:id", async (req, res) => {
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