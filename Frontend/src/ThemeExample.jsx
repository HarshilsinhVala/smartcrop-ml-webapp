import React from 'react';
import { useTheme } from './ThemeContext';
import './ThemeExample.css';

/**
 * This is an example component showing how to use the theme system.
 * You can refer to this component when creating new themed components.
 */

function ThemeExample() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="theme-example-container">
      <div className="example-card">
        <h2>Theme Example Component</h2>
        <p>Current Theme: <strong>{theme}</strong></p>
        
        <div className="example-section">
          <h3>Using CSS Variables</h3>
          <p>All colors in this component use CSS variables defined in index.css.</p>
          <p>They automatically change when you switch themes!</p>
        </div>

        <div className="example-section">
          <h3>Using useTheme Hook</h3>
          <p>You can access the current theme in JavaScript:</p>
          <pre>
            const {'{theme, toggleTheme}'} = useTheme();
          </pre>
        </div>

        <div className="example-buttons">
          <button className="example-btn btn-primary">Primary Button</button>
          <button className="example-btn btn-secondary">Secondary Button</button>
          <button className="example-btn" onClick={toggleTheme}>
            Toggle Theme
          </button>
        </div>

        <div className="theme-colors">
          <h3>Available CSS Variables:</h3>
          <div className="color-grid">
            <div className="color-item">
              <div className="color-box" style={{backgroundColor: 'var(--bg-color)'}}></div>
              <span>--bg-color</span>
            </div>
            <div className="color-item">
              <div className="color-box" style={{backgroundColor: 'var(--card-bg)'}}></div>
              <span>--card-bg</span>
            </div>
            <div className="color-item">
              <div className="color-box" style={{backgroundColor: 'var(--primary-color)'}}></div>
              <span>--primary-color</span>
            </div>
            <div className="color-item">
              <div className="color-box" style={{backgroundColor: 'var(--secondary-bg)'}}></div>
              <span>--secondary-bg</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ThemeExample;
