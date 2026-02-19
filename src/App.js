import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import Hero from './Hero';
import SmartCropSection from './SmartCropSection';
import LoginModal from './LoginModal';
import RegisterModal from './RegisterModal';
import Auction from './Auction';
import CropPrediction from './CropPrediction';
import DiseaseDetection from './DiseaseDetection';
import './App.css';

function App() {
  const [modalType, setModalType] = useState(null);

  const handleOpenModal = (type) => setModalType(type);
  const handleCloseModal = () => setModalType(null);

  return (
    <Router>
      <div className="App">
        <Navbar onLoginClick={() => handleOpenModal('login')} onRegisterClick={() => handleOpenModal('register')} />

        <Routes>
          <Route path="/" element={<><Hero /><SmartCropSection /></>} />
          <Route path="/auction" element={<Auction />} />
          <Route path="/crop-prediction" element={<CropPrediction />} />
          <Route path="/disease-detection" element={<DiseaseDetection />} />
        </Routes>
        

        {modalType === 'login' && (   
          <LoginModal isOpen={modalType === 'login'} onClose={handleCloseModal} onSwitchToRegister={() => handleOpenModal('register')} />
        )}
        {modalType === 'register' && (
          <RegisterModal isOpen={modalType === 'register'} onClose={handleCloseModal} onSwitchToLogin={() => handleOpenModal('login')} />
        )}
      </div>
    </Router>
  );
}

export default App;