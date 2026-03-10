const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/User");
const authMiddleware = require("../middleware/authMiddleware");
require("dotenv").config();

const router = express.Router();

// Register User
router.post("/register", async (req, res) => {
  try {
    console.log("➡ Register API called");
    console.log("📩 Received Data:", req.body);

    const { username, email, password } = req.body;

    if (!username || !email || !password) {
      console.log("❌ Missing fields:", { username, email, password });
      return res.status(400).json({ error: "All fields are required" });
    }

    const userExists = await User.findOne({ $or: [{ username }, { email }] });
    if (userExists) {
      console.log("❌ User already exists:", userExists);
      return res.status(400).json({ error: "Username or email already exists" });
    }

    // Let the User model's pre-save hook hash the password to avoid double-hashing
    const newUser = new User({ username, email, password });
    await newUser.save();

    console.log("✅ User registered successfully:", newUser);
    res.status(201).json({ message: "✅ User registered successfully" });
  } catch (error) {
    console.error("❌ Registration Error:", error);
    // Handle duplicate key errors from MongoDB more clearly
    if (error.code === 11000) {
      return res.status(400).json({ error: "Username or email already exists" });
    }
    res.status(500).json({ error: "❌ Server error" });
  }
});

// Login User
router.post("/login", async (req, res) => {
  try {
    console.log("➡ Login API called");
    console.log("📩 Received Data:", req.body);

    const { username, password } = req.body;

    if (!username || !password) {
      console.log("❌ Missing fields:", { username, password });
      return res.status(400).json({ error: "All fields are required" });
    }

    // Fetch user from DB
    const user = await User.findOne({ username });
    if (!user) {
      console.log("❌ User not found:", username);
      return res.status(400).json({ error: "Invalid username or password" });
    }

    // Debugging: Print stored and entered passwords
    console.log("🔍 Stored Hashed Password:", user.password);
    console.log("🔍 Entered Password:", password);

    // Check password
    const isMatch = await bcrypt.compare(password, user.password);
    console.log("🔍 Password Match Result:", isMatch);

    if (!isMatch) {
      console.log("❌ Incorrect password for:", username);
      return res.status(400).json({ error: "Invalid username or password" });
    }

    // Generate JWT token
    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "1h" });
    console.log("✅ Login successful for:", username);
    res.json({ message: "✅ Login successful", token });
  } catch (error) {
    console.error("❌ Login Error:", error);
    res.status(500).json({ error: "❌ Server error" });
  }
});

// DEBUG: List all users (no password) - remove in production
router.get('/users', async (req, res) => {
  try {
    const users = await User.find({}, { password: 0 });
    res.json({ users });
  } catch (err) {
    console.error('❌ Error fetching users:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Get current user info
router.get('/me', authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-password');
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json({ user });
  } catch (err) {
    console.error('❌ Error fetching user:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;