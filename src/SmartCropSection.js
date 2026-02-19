import React from 'react';
import './SmartCropSection.css';



function SmartCropSection() {
  return (
    <div className="smart-crop-section">
      <div className="card">
        <img
          src="https://www.pngarts.com/files/3/Cartoon-Farmer-PNG-Photo.png"
          alt="Farmer with rake"
          className="card-image"
        />
        <div className="card-content1">
          <h2>Smart Crop Recommendations</h2>
          <p>
            Unlock the full potential of your farm with personalized crop</p>
            <p>recommendations based on soil type, climate, and other factors.Maximize<p/>
            <p>your yield and profitability with expert advice tailored to
            your unique</p> <p>farming conditions.</p>
          </p>
        </div>
      </div>
      
      <div className="card">
        <img
          src="https://png.pngtree.com/png-vector/20240325/ourmid/pngtree-earth-day-with-a-symbol-of-growth-and-unity-png-image_12229437.png"
          alt="Farmer with rake"
          className="card-image"
        />
        <div className="card-content1">
          <h2>Plant Health Guardian</h2>
          <p>
          Ensure the well-being of your crops by detecting and addressing plant</p>
            <p>diseases in real-time. Simply capture a photo of the affected plant, and our<p/>
            <p>advanced system will analyze and provide effective solutions to keep your</p> <p>farm thriving.</p>
          </p>
        </div>
      </div>
      <div className="card">
        <img
          src="https://cdn3d.iconscout.com/3d/premium/thumb/hand-holding-auction-bid-board-3d-illustration-download-in-png-blend-fbx-gltf-file-formats--bidding-raise-sign-business-illustrations-3932217.png"
          alt="Farmer with rake"
          className="card-image"
        />
        <div className="card-content1">
          <h2>Crop Marketplace</h2>
          <p>
          Maximize your crop sales with our integrated auction platform. Showcase</p>
            <p>your harvest to a wide audience of buyers, and let them bid for the best<p/>
            <p>quality produce. Experience a transparent and efficient way to sell your
            </p> <p>crops while getting the best value for your hard work.</p>
          </p>
        </div>
      </div>
      
    </div>
  );
}

export default SmartCropSection;
