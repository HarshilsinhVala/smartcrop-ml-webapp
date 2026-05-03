import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import Navbar from './Navbar';
import Hero from './Hero';
import SmartCropSection from './SmartCropSection';
import LoginModal from './LoginModal';
import RegisterModal from './RegisterModal';
import Auction from './Auction';
import CropPrediction from './CropPrediction';
import DiseaseDetection from './DiseaseDetection';
import AdminDashboard from './AdminDashboard';
import { ThemeProvider } from './ThemeContext';
import './App.css';

const API_BASE = 'http://localhost:3001';

function App() {
  const [modalType, setModalType] = useState(null);
  const [user, setUser]           = useState(null);
  const [adminUser, setAdminUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('adminUser')); } catch { return null; }
  });

  const handleOpenModal  = (type) => setModalType(type);
  const handleCloseModal = () => setModalType(null);

  // ── Fetch logged-in regular user ──────────────────────────
  const fetchUser = async () => {
    const token = localStorage.getItem('token');
    if (!token) return;
    try {
      const res = await axios.get(`${API_BASE}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUser(res.data.user);
    } catch {
      localStorage.removeItem('token');
      setUser(null);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  // ── Admin callbacks ───────────────────────────────────────
  const handleAdminLoginSuccess = (admin) => {
    setAdminUser(admin);
  };

  const handleAdminLogout = () => {
    localStorage.removeItem('adminToken');
    localStorage.removeItem('adminUser');
    setAdminUser(null);
  };

  useEffect(() => { fetchUser(); }, []);

  return (
    <ThemeProvider>
      <Router>
        <div className="App">
          {/* Navbar — hidden on admin dashboard */}
          <Routes>
            <Route path="/admin/*" element={null} />
            <Route
              path="*"
              element={
                <Navbar
                  onLoginClick={() => handleOpenModal('login')}
                  onRegisterClick={() => handleOpenModal('register')}
                  user={user}
                  adminUser={adminUser}
                  onLogout={handleLogout}
                  onAdminLogout={handleAdminLogout}
                />
              }
            />
          </Routes>

          <Routes>
            {/* Public routes */}
            <Route path="/" element={<><Hero /><SmartCropSection /></>} />
            <Route path="/auction"          element={<Auction />} />
            <Route path="/crop-prediction"  element={<CropPrediction />} />
            <Route path="/disease-detection" element={<DiseaseDetection />} />

            {/* Admin route — protected */}
            <Route
              path="/admin"
              element={
                adminUser
                  ? <AdminDashboard onAdminLogout={handleAdminLogout} />
                  : <Navigate to="/" replace />
              }
            />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>

          {/* Modals */}
          {modalType === 'login' && (
            <LoginModal
              isOpen
              onClose={handleCloseModal}
              onSwitchToRegister={() => handleOpenModal('register')}
              onLoginSuccess={fetchUser}
              onAdminLoginSuccess={(admin) => {
                handleAdminLoginSuccess(admin);
                handleCloseModal();
              }}
            />
          )}
          {modalType === 'register' && (
            <RegisterModal
              isOpen
              onClose={handleCloseModal}
              onSwitchToLogin={() => handleOpenModal('login')}
            />
          )}
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;