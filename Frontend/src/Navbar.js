import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useTheme } from './ThemeContext';
import './Navbar.css';

function Navbar({ onLoginClick, onRegisterClick, user, adminUser, onLogout, onAdminLogout }) {
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
  };

  const handleAdminLogout = () => {
    if (typeof onAdminLogout === 'function') onAdminLogout();
  };

  return (
    <nav className="navbar">
      {/* Logo */}
      <div className="logo">
        <a href="/">
          <img
            src="https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcQtu7mLs8RW2MDI6Em0lA-cDWGQ2h-8PK6rNyawCNF3u02Cmokw"
            alt="Farmer Logo"
            id="farmer-img"
          />
        </a>
      </div>

      {/* Theme toggle */}
      <div className="theme-toggle">
        <button
          className="theme-btn"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          aria-label="Toggle theme"
          id="theme-toggle-btn"
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </div>

      {/* Nav links */}
      <ul className="nav-links">
        <li>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active-link' : ''}>
            Home
          </NavLink>
        </li>
        <li>
          <NavLink to="/auction" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Auction
          </NavLink>
        </li>
        <li>
          <NavLink to="/crop-prediction" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Crop Prediction
          </NavLink>
        </li>
        <li>
          <NavLink to="/disease-detection" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Disease Detection
          </NavLink>
        </li>

        {/* Admin dashboard link — visible only when admin is logged in */}
        {adminUser && (
          <li>
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                'admin-nav-link' + (isActive ? ' active-link' : '')
              }
              id="admin-dashboard-nav-link"
            >
              🛡️ Admin Dashboard
            </NavLink>
          </li>
        )}
      </ul>

      {/* Auth area */}
      <div className="auth-buttons">
        {/* Admin badge + logout */}
        {adminUser && (
          <div className="user-info">
            <span className="admin-chip" id="admin-chip">
              🛡️ {adminUser.username || 'Admin'}
            </span>
            <button
              className="auth-btn logout-btn"
              onClick={handleAdminLogout}
              id="admin-logout-navbar-btn"
            >
              Admin Logout
            </button>
          </div>
        )}

        {/* Regular user */}
        {user && !adminUser && (
          <div className="user-info">
            <span id="user-welcome-text">Welcome, {user.username}!</span>
            <button
              className="auth-btn logout-btn"
              onClick={handleLogout}
              id="user-logout-btn"
            >
              Logout
            </button>
          </div>
        )}

        {/* Not logged in at all */}
        {!user && !adminUser && (
          <>
            <button
              className="auth-btn login-btn"
              onClick={onLoginClick}
              id="navbar-login-btn"
            >
              Login
            </button>
            <button
              className="auth-btn register-btn"
              onClick={onRegisterClick}
              id="navbar-register-btn"
            >
              Register
            </button>
          </>
        )}
      </div>
    </nav>
  );
}

export default Navbar;