import React, { useState } from "react";
import axios from "axios";
import { FaUser, FaLock, FaShieldAlt } from "react-icons/fa";
import "./LoginModal.css";

const API_BASE = "http://localhost:3001";

function LoginModal({ isOpen, onClose, onSwitchToRegister, onLoginSuccess, onAdminLoginSuccess }) {
  const [tab, setTab] = useState("user"); // "user" | "admin"
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError(null);
  };

  const switchTab = (t) => {
    setTab(t);
    setError(null);
    setFormData({ username: "", password: "" });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (tab === "user") {
        const response = await axios.post(`${API_BASE}/auth/login`, formData, {
          headers: { "Content-Type": "application/json" },
        });
        localStorage.setItem("token", response.data.token);
        setFormData({ username: "", password: "" });
        if (typeof onLoginSuccess === "function") onLoginSuccess();
        if (typeof onClose === "function") onClose();
      } else {
        const response = await axios.post(`${API_BASE}/admin/login`, formData, {
          headers: { "Content-Type": "application/json" },
        });
        localStorage.setItem("adminToken", response.data.token);
        localStorage.setItem("adminUser", JSON.stringify(response.data.user));
        setFormData({ username: "", password: "" });
        if (typeof onAdminLoginSuccess === "function") onAdminLoginSuccess(response.data.user);
        if (typeof onClose === "function") onClose();
      }
    } catch (err) {
      setError(err.response?.data?.error || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="login-container">
        {/* Left panel */}
        <div className="left-section">
          {tab === "admin" ? (
            <>
              <FaShieldAlt size={42} style={{ marginBottom: 12, opacity: 0.9 }} />
              <h2>Admin Portal</h2>
              <p>Access the SmartCrop admin dashboard to manage users, auctions, and platform data.</p>
            </>
          ) : (
            <>
              <h2>Hello, Welcome!</h2>
              <p>Don't have an account?</p>
              <button className="register-button" onClick={onSwitchToRegister}>
                Register
              </button>
            </>
          )}
        </div>

        {/* Right panel */}
        <div className="right-section">
          <button className="close-btn" onClick={onClose} aria-label="Close">
            ×
          </button>

          {/* Tab switcher */}
          <div className="login-tabs" role="tablist">
            <button
              className={`login-tab${tab === "user" ? " active" : ""}`}
              onClick={() => switchTab("user")}
              role="tab"
              aria-selected={tab === "user"}
              id="tab-user"
            >
              👤 User Login
            </button>
            <button
              className={`login-tab${tab === "admin" ? " active" : ""}`}
              onClick={() => switchTab("admin")}
              role="tab"
              aria-selected={tab === "admin"}
              id="tab-admin"
            >
              🛡️ Admin Login
            </button>
          </div>

          {tab === "admin" && (
            <p className="admin-login-hint">
              Admin credentials are set in the server environment variables.
            </p>
          )}

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <FaUser className="icon" />
              <input
                type="text"
                name="username"
                placeholder={tab === "admin" ? "Admin Username" : "Username"}
                value={formData.username}
                onChange={handleChange}
                required
                autoComplete="username"
                id={`${tab}-username-input`}
              />
            </div>

            <div className="input-group">
              <FaLock className="icon" />
              <input
                type="password"
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
                required
                autoComplete="current-password"
                id={`${tab}-password-input`}
              />
            </div>

            <button
              type="submit"
              className={`login-button${tab === "admin" ? " admin-submit-btn" : ""}`}
              disabled={loading}
              id={`${tab}-login-submit`}
            >
              {loading ? "Signing in…" : tab === "admin" ? "🛡️ Admin Sign In" : "Login"}
            </button>

            {error && <p className="error" style={{ marginTop: 12 }}>{error}</p>}
          </form>
        </div>
      </div>
    </div>
  );
}

export default LoginModal;