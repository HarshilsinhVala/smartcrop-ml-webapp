const express  = require("express");
const jwt      = require("jsonwebtoken");
const bcrypt   = require("bcryptjs");
const router   = express.Router();

// Import your existing User model — adjust path if different
const User    = require("../models/User");
// Import or create a simple Auction model
const Auction = require("../models/Auction");

// ── Admin credentials from .env ──
// Add these to your .env file:
//   ADMIN_USERNAME=admin
//   ADMIN_PASSWORD=admin@kisaan123
//   JWT_SECRET=your_jwt_secret

// ── Middleware: verify JWT and check admin role ──
function verifyAdmin(req, res, next) {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) {
    return res.status(401).json({ error: "No token provided" });
  }
  try {
    const decoded = jwt.verify(auth.split(" ")[1], process.env.JWT_SECRET);
    if (decoded.role !== "admin") {
      return res.status(403).json({ error: "Admin access required" });
    }
    req.admin = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}

// ════════════════════════════════════════
// POST /admin/login  — Admin login
// ════════════════════════════════════════
router.post("/login", async (req, res) => {
  const { username, password } = req.body;

  const adminUsername = process.env.ADMIN_USERNAME || "admin";
  const adminPassword = process.env.ADMIN_PASSWORD || "admin@kisaan123";

  if (username !== adminUsername || password !== adminPassword) {
    return res.status(401).json({ error: "Invalid admin credentials" });
  }

  const token = jwt.sign(
    { username, role: "admin" },
    process.env.JWT_SECRET,
    { expiresIn: "8h" }
  );

  res.json({
    token,
    user: { username, role: "admin" },
  });
});

// ════════════════════════════════════════
// GET /admin/users  — Get all users
// ════════════════════════════════════════
router.get("/users", verifyAdmin, async (req, res) => {
  try {
    const users = await User.find({}, "-password").sort({ createdAt: -1 });
    res.json({ users });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch users" });
  }
});

// ════════════════════════════════════════
// DELETE /admin/users/:id  — Delete a user
// ════════════════════════════════════════
router.delete("/users/:id", verifyAdmin, async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) return res.status(404).json({ error: "User not found" });
    if (user.role === "admin") return res.status(403).json({ error: "Cannot delete admin user" });

    await User.findByIdAndDelete(req.params.id);
    res.json({ message: "User deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete user" });
  }
});

// ════════════════════════════════════════
// GET /admin/auctions  — Get all auctions
// ════════════════════════════════════════
router.get("/auctions", verifyAdmin, async (req, res) => {
  try {
    const auctions = await Auction.find().sort({ createdAt: -1 });
    res.json({ auctions });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch auctions" });
  }
});

// ════════════════════════════════════════
// POST /admin/auctions  — Add new auction
// ════════════════════════════════════════
router.post("/auctions", verifyAdmin, async (req, res) => {
  try {
    const { name, description, quantity, price } = req.body;
    if (!name || !description) {
      return res.status(400).json({ error: "Name and description are required" });
    }
    const auction = new Auction({ name, description, quantity, price });
    await auction.save();
    res.status(201).json({ auction });
  } catch (err) {
    res.status(500).json({ error: "Failed to add auction" });
  }
});

// ════════════════════════════════════════
// DELETE /admin/auctions/:id  — Remove auction
// ════════════════════════════════════════
router.delete("/auctions/:id", verifyAdmin, async (req, res) => {
  try {
    const auction = await Auction.findByIdAndDelete(req.params.id);
    if (!auction) return res.status(404).json({ error: "Auction not found" });
    res.json({ message: "Auction removed successfully" });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete auction" });
  }
});

module.exports = router;
