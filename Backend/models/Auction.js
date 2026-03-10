const mongoose = require("mongoose");

const AuctionSchema = new mongoose.Schema({
  name: String,
  description: String,
  quantity: Number,
  price: Number,
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Auction", AuctionSchema);