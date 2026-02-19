import React, { useState } from "react";
import axios from "axios";
import { FaUser, FaLock, FaGoogle, FaFacebookF, FaGithub, FaLinkedinIn } from "react-icons/fa";
import "./LoginModal.css";

function LoginModal({ isOpen, onClose, onSwitchToRegister }) {
  const [formData, setFormData] = useState({ username: "", password: "" });
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
      const response = await axios.post("http://localhost:5000/auth/login", formData, {
        headers: { "Content-Type": "application/json" },
      });

      setMessage(response.data.message);
      localStorage.setItem("token", response.data.token); // Save token for authentication
      setFormData({ username: "", password: "" }); // Clear input fields after login
        // Close the modal after successful login
        if (typeof onClose === "function") onClose();
    } catch (err) {
      setError(err.response?.data?.error || "Login failed");
    }
  };

  return (
    <div className="modal-overlay">
      <div className="login-container">
        <div className="left-section">
          <h2>Hello, Welcome!</h2>
          <p>Don't have an account?</p>
          <button className="register-button" onClick={onSwitchToRegister}>Register</button>
        </div>

        <div className="right-section">
          <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
          <h2>Login</h2>
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <FaUser className="icon" />
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
              <FaLock className="icon" />
              <input 
                type="password" 
                name="password" 
                placeholder="Password" 
                value={formData.password} 
                onChange={handleChange} 
                required 
              />
            </div>

            <button type="submit" className="login-button">Login</button>

            {message && <p className="success">{message}</p>}
            {error && <p className="error">{error}</p>}
          </form>
        </div>
      </div>
    </div>
  );
}

export default LoginModal;