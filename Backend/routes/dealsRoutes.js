const express = require("express");
const Deal = require("../models/Deal");

const router = express.Router();

// Create a new deal
router.post("/", async (req, res) => {
  try {
    const { auctionId, name, quantity, price } = req.body;
    const newDeal = new Deal({ auctionId, name, quantity, price });
    await newDeal.save();
    res.json({ message: "Deal created successfully", deal: newDeal });
  } catch (error) {
    res.status(500).json({ error: "Error creating deal", details: error });
  }
});

// Get all deals
router.get("/", async (req, res) => {
  try {
    const deals = await Deal.find().populate("auctionId");
    res.json(deals);
  } catch (error) {
    res.status(500).json({ error: "Error fetching deals", details: error });
  }
});

module.exports = router;