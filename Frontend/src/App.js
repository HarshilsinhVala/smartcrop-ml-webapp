import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import Navbar from './Navbar';
import Hero from './Hero';
import SmartCropSection from './SmartCropSection';
import LoginModal from './LoginModal';
import RegisterModal from './RegisterModal';
import Auction from './Auction';
import CropPrediction from './CropPrediction';
import DiseaseDetection from './DiseaseDetection';
import { ThemeProvider } from './ThemeContext';
import './App.css';

function App() {
  const [modalType, setModalType] = useState(null);
  const [user, setUser] = useState(null);

  const handleOpenModal = (type) => setModalType(type);
  const handleCloseModal = () => setModalType(null);

  const fetchUser = async () => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const response = await axios.get('http://localhost:3001/auth/me', {
          headers: { Authorization: `Bearer ${token}` },
        });
        setUser(response.data.user);
      } catch (err) {
        console.error('Failed to fetch user:', err);
        localStorage.removeItem('token');
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  useEffect(() => {
    fetchUser();
  }, []);

  return (
    <ThemeProvider>
      <Router>
        <div className="App">
          <Navbar 
            onLoginClick={() => handleOpenModal('login')} 
            onRegisterClick={() => handleOpenModal('register')} 
            user={user} 
            onLogout={handleLogout} 
          />

          <Routes>
            <Route path="/" element={<><Hero /><SmartCropSection /></>} />
            <Route path="/auction" element={<Auction />} />
            <Route path="/crop-prediction" element={<CropPrediction />} />
            <Route path="/disease-detection" element={<DiseaseDetection />} />
          </Routes>
          

          {modalType === 'login' && (   
            <LoginModal isOpen={modalType === 'login'} onClose={handleCloseModal} onSwitchToRegister={() => handleOpenModal('register')} onLoginSuccess={fetchUser} />
          )}
          {modalType === 'register' && (
            <RegisterModal isOpen={modalType === 'register'} onClose={handleCloseModal} onSwitchToLogin={() => handleOpenModal('login')} />
          )}
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;