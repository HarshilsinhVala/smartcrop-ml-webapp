import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css';

function Navbar({ onLoginClick, onRegisterClick }) {
  return (
    <nav className="navbar">
      <div className="logo">
        <a href="/">
          <img
            src="https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcQtu7mLs8RW2MDI6Em0lA-cDWGQ2h-8PK6rNyawCNF3u02Cmokw"
            alt="Farmer Logo"
            id="farmer-img"
          />
        </a>
      </div>

      <ul className="nav-links">
      
        <li>
          <NavLink to="/" exact activeClassName="active-link">
            Home
          </NavLink>
        </li>
        <li>
          <NavLink to="/auction" activeClassName="active-link">
            Auction
          </NavLink>
        </li>
        <li>
          <NavLink to="/crop-prediction" activeClassName="active-link">
            Crop Prediction
          </NavLink>
        </li>
        <li>
          <NavLink to="/disease-detection" activeClassName="active-link">
            Disease Detection
          </NavLink>
        </li>

      </ul>

      <div className="auth-buttons">
        <button className="auth-btn login-btn" onClick={onLoginClick}>
          Login
        </button>
        <button className="auth-btn register-btn" onClick={onRegisterClick}>
          Register
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
