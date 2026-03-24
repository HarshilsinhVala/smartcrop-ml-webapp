const mongoose = require("mongoose");

const auctionSchema = new mongoose.Schema(
  {
    name:        { type: String, required: true, trim: true },
    description: { type: String, required: true, trim: true },
    quantity:    { type: Number, default: null },
    price:       { type: Number, default: null },
    createdBy:   { type: String, default: "admin" },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Auction", auctionSchema);