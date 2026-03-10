import React from 'react';
import { NavLink } from 'react-router-dom';
import { useTheme } from './ThemeContext';
import './Navbar.css';

function Navbar({ onLoginClick, onRegisterClick, user, onLogout }) {
  const { theme, toggleTheme } = useTheme();
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

      <div className="theme-toggle">
        <button 
          className="theme-btn" 
          onClick={toggleTheme}
          title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          aria-label="Toggle theme"
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
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
        {user ? (
          <div className="user-info">
            <span>Welcome, {user.username}!</span>
            <button className="auth-btn logout-btn" onClick={onLogout}>
              Logout
            </button>
          </div>
        ) : (
          <>
            <button className="auth-btn login-btn" onClick={onLoginClick}>
              Login
            </button>
            <button className="auth-btn register-btn" onClick={onRegisterClick}>
              Register
            </button>
          </>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
