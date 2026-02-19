import React, { useState } from "react";
import axios from "axios";
import "./RegisterModal.css";

function RegisterModal({ isOpen, onClose, onSwitchToLogin }) {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);

  if (!isOpen) return null;

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setMessage(null);

    try {
      const response = await axios.post("http://localhost:5000/auth/register", formData, {
        headers: { "Content-Type": "application/json" },
      });

      setMessage(response.data.message);
      setFormData({ username: "", email: "", password: "" }); // Clear fields after successful registration
    } catch (err) {
      setError(err.response?.data?.error || "Registration failed");
    }
  };

  return (
    <div className="modal-overlay">
      <div className="register-container">
        <div className="left-section">
          <h2>Welcome Back!</h2>
          <p>Already have an account?</p>
          <button className="login-button1" onClick={onSwitchToLogin}>Login</button>
        </div>

        <div className="right-section">
          <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
          <h2>Registration</h2>
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <input
                type="text"
                name="username"
                placeholder="Username"
                value={formData.username}
                onChange={handleChange}
                required
              />
            </div>

            <div className="input-group">
              <input
                type="email"
                name="email"
                placeholder="Email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>

            <div className="input-group">
              <input
                type="password"
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
                required
              />
            </div>

            <button type="submit" className="register-btn1">Register</button>

            {message && <p className="success">{message}</p>}
            {error && <p className="error">{error}</p>}
          </form>
        </div>
      </div>
    </div>
  );
}

export default RegisterModal;