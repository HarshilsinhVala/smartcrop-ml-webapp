import React, { useState } from "react";
import "./Auction.css";

function Auction() {
  const [cards, setCards] = useState([
    {
      name: "Rice",
      description: "A staple grain rich in carbs, widely consumed globally.",
      quantity: "",
      price: "",
    },
    {
      name: "Wheat",
      description: "Used for flour, bread, and pasta; high in fiber and protein.",
      quantity: "",
      price: "",
    },
    {
      name: "Oats",
      description: "A heart-healthy grain, great for lowering cholesterol.",
      quantity: "",
      price: "",
    },
    {
      name: "Millets ",
      description: "Nutrient-rich, gluten-free grains with high fiber and protein.",
      quantity: "",
      price: "",
    },
  ]);

  const [newCard, setNewCard] = useState({
    name: "",
    description: "",
    quantity: "",
    price: "",
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewCard({ ...newCard, [name]: value });
  };

  const handleCardFieldChange = (index, field, value) => {
    const updatedCards = [...cards];
    updatedCards[index][field] = value;
    setCards(updatedCards);
  };

  const addCard = () => {
    if (newCard.name.trim() && newCard.description.trim()) {
      setCards([...cards, { ...newCard }]);
      setNewCard({ name: "", description: "", quantity: "", price: "" }); // Reset form
    } else {
      alert("Please fill out both fields.");
    }
  };

  const removeCard = (index) => {
    const updatedCards = cards.filter((_, cardIndex) => cardIndex !== index);
    setCards(updatedCards);
  };

  const makeDeal = (index) => {
    const card = cards[index];
    if (!card.quantity || !card.price) {
      alert("Please enter quantity and price to make the deal.");
      return;
    }
    alert(
      `Deal confirmed for ${card.name}!\nQuantity: ${card.quantity}\nPrice: $${card.price}`
    );
  };

  return (
    <div className="auction-page">
      <h1>Live Auctions</h1>
      <div className="auction-grid">
        {cards.map((card, index) => (
          <div className="auction-card" key={index}>
            <h2>Name: {card.name}</h2>
            <p>{card.description}</p>
            <div className="card-input-group">
              <input
                type="number"
                placeholder="Enter quantity"
                value={card.quantity}
                onChange={(e) =>
                  handleCardFieldChange(index, "quantity", e.target.value)
                }
              />
              <input
                type="number"
                placeholder="Enter price"
                value={card.price}
                onChange={(e) =>
                  handleCardFieldChange(index, "price", e.target.value)
                }
              />
            </div>
            <button
              className="make-deal-button"
              onClick={() => makeDeal(index)}
            >
              Make Deal
            </button>
            <button
              className="remove-card-button"
              onClick={() => removeCard(index)}
            >
              Remove
            </button>
          </div>
        ))}
      </div>
      <div className="add-card-form">
        <input
          type="text"
          name="name"
          value={newCard.name}
          placeholder="Enter name"
          onChange={handleInputChange}
        />
        <textarea
          name="description"
          value={newCard.description}
          placeholder="Enter description"
          onChange={handleInputChange}
        />
        <button className="add-card-button" onClick={addCard}>
          +
        </button>
      </div>
    </div>
  );
}

export default Auction;
  