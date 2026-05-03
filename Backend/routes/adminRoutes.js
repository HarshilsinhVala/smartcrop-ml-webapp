const express  = require("express");
const jwt      = require("jsonwebtoken");
const router   = express.Router();

const User    = require("../models/User");
const Auction = require("../models/Auction");

function requireEnv(name) {
  const v = process.env[name];
  return typeof v === "string" && v.trim().length > 0 ? v : null;
}

// ── Middleware: verify JWT and check admin role ──────────────
function verifyAdmin(req, res, next) {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) {
    return res.status(401).json({ error: "No token provided" });
  }
  try {
    const jwtSecret = requireEnv("JWT_SECRET");
    if (!jwtSecret) {
      return res.status(500).json({ error: "Server misconfigured: JWT_SECRET missing" });
    }
    const decoded = jwt.verify(auth.split(" ")[1], jwtSecret);
    if (decoded.role !== "admin") {
      return res.status(403).json({ error: "Admin access required" });
    }
    req.admin = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}

// ════════════════════════════════════════════════════════════
// POST /admin/login
// ════════════════════════════════════════════════════════════
router.post("/login", async (req, res) => {
  const { username, password } = req.body;

  const adminUsername = requireEnv("ADMIN_USERNAME");
  const adminPassword = requireEnv("ADMIN_PASSWORD");
  const jwtSecret     = requireEnv("JWT_SECRET");

  if (!adminUsername || !adminPassword || !jwtSecret) {
    return res.status(500).json({ error: "Server misconfigured: missing admin env vars" });
  }

  if (username !== adminUsername || password !== adminPassword) {
    return res.status(401).json({ error: "Invalid admin credentials" });
  }

  const token = jwt.sign(
    { username, role: "admin" },
    jwtSecret,
    { expiresIn: "8h" }
  );

  res.json({
    token,
    user: { username, role: "admin" },
  });
});

// ════════════════════════════════════════════════════════════
// GET /admin/stats  — Dashboard overview stats
// ════════════════════════════════════════════════════════════
router.get("/stats", verifyAdmin, async (req, res) => {
  try {
    const [totalUsers, totalAuctions, auctions] = await Promise.all([
      User.countDocuments(),
      Auction.countDocuments(),
      Auction.find({}, "price"),
    ]);

    const totalValue = auctions.reduce((sum, a) => sum + (a.price || 0), 0);
    const avgPrice   = totalAuctions > 0 ? Math.round(totalValue / totalAuctions) : 0;
    const activeListings = auctions.filter((a) => a.price > 0).length;

    // Users registered in the last 7 days
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const newUsers = await User.countDocuments({ createdAt: { $gte: sevenDaysAgo } });

    res.json({ totalUsers, totalAuctions, activeListings, avgPrice, newUsers });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch stats" });
  }
});

// ════════════════════════════════════════════════════════════
// GET /admin/users  — Get all users (sorted newest first)
// ════════════════════════════════════════════════════════════
router.get("/users", verifyAdmin, async (req, res) => {
  try {
    const users = await User.find({}, "-password").sort({ createdAt: -1 });
    res.json({ users });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch users" });
  }
});

// ════════════════════════════════════════════════════════════
// DELETE /admin/users/:id  — Delete a user
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
// PATCH /admin/users/:id/role  — Toggle user role
// ════════════════════════════════════════════════════════════
router.patch("/users/:id/role", verifyAdmin, async (req, res) => {
  try {
    const { role } = req.body;
    if (!["user", "admin"].includes(role)) {
      return res.status(400).json({ error: "Invalid role. Must be 'user' or 'admin'" });
    }
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { role },
      { new: true, select: "-password" }
    );
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json({ user });
  } catch (err) {
    res.status(500).json({ error: "Failed to update user role" });
  }
});

// ════════════════════════════════════════════════════════════
// GET /admin/auctions  — Get all auctions
// ════════════════════════════════════════════════════════════
router.get("/auctions", verifyAdmin, async (req, res) => {
  try {
    const auctions = await Auction.find().sort({ createdAt: -1 });
    res.json({ auctions });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch auctions" });
  }
});

// ════════════════════════════════════════════════════════════
// POST /admin/auctions  — Add new auction
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
// DELETE /admin/auctions/:id  — Remove auction
// ════════════════════════════════════════════════════════════
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
