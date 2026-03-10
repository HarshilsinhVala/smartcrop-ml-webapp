# Theme Configuration Guide

## How to Use Dark Mode & Light Mode

Your app now supports **Light Mode** and **Dark Mode** with automatic theme switching!

### Features:
- 🌙 Click the moon/sun icon in the navbar to switch themes
- 💾 Theme preference is saved in localStorage (persists across sessions)
- 🎨 Smooth transitions between themes
- 📱 Works on all screen sizes

---

## CSS Variables Reference

The following CSS variables are used throughout the app and automatically change based on the selected theme:

### Colors:
```css
--bg-color: Background color (body/pages)
--text-color: Primary text color
--navbar-bg: Navbar background
--navbar-text: Navbar text color
--primary-color: Main accent color (buttons, links)
--secondary-bg: Secondary background (cards, sections)
--card-bg: Card/component background
--card-border: Card border color
--shadow-color: Shadow colors
--input-bg: Input field background
--input-border: Input field border
--button-bg: Button background
--button-hover: Button hover state
```

### Light Mode (Default):
- Background: White (#ffffff)
- Text: Black (#000000)
- Navbar: Black (#000000)
- Primary: Blue (#007bff)
- Cards: White with light borders

### Dark Mode:
- Background: Dark Grey (#1a1a1a)
- Text: Light Grey (#e0e0e0)
- Navbar: Very Dark (#121212)
- Primary: Bright Blue (#4A90E2)
- Cards: Dark Grey with darker borders

---

## How to Apply Theme to Components

### Option 1: Using CSS Variables (Recommended)

In your CSS files, use `var()` to reference theme colors:

```css
/* In your component CSS file */
.my-component {
  background-color: var(--card-bg);
  color: var(--text-color);
  border: 1px solid var(--card-border);
  box-shadow: 0 2px 4px var(--shadow-color);
  transition: background-color 0.3s ease, color 0.3s ease;
}

.my-component:hover {
  background-color: var(--primary-color);
  color: var(--navbar-bg);
}
```

### Option 2: Using useTheme Hook in JavaScript

```javascript
import { useTheme } from './ThemeContext';

function MyComponent() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div>
      <p>Current theme: {theme}</p>
      <button onClick={toggleTheme}>
        Switch to {theme === 'light' ? 'dark' : 'light'} mode
      </button>
    </div>
  );
}
```

---

## Updating Existing Components

To theme-enable an existing component:

1. **Replace hardcoded colors** with CSS variables:

   **Before:**
   ```css
   .card {
     background-color: #ffffff;
     color: #000000;
     border: 1px solid #ddd;
   }
   ```

   **After:**
   ```css
   .card {
     background-color: var(--card-bg);
     color: var(--text-color);
     border: 1px solid var(--card-border);
     transition: all 0.3s ease;
   }
   ```

2. **Add transitions** for smooth theme switching:
   ```css
   .my-element {
     transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
   }
   ```

---

## Customizing the Theme

To change colors, edit the CSS variables in `index.css`:

```css
/* In src/index.css */
:root[data-theme="light"] {
  --primary-color: #ff6b6b; /* Change primary color */
  --navbar-bg: #2c3e50;     /* Change navbar */
}

:root[data-theme="dark"] {
  --primary-color: #ff9500; /* Dark mode primary */
  --navbar-bg: #1a1a2e;     /* Dark mode navbar */
}
```

---

## Files Modified:

✅ **Created:** `src/ThemeContext.js` - Theme provider and hook
✅ **Updated:** `src/App.js` - Wrapped with ThemeProvider
✅ **Updated:** `src/Navbar.js` - Added theme toggle button
✅ **Updated:** `src/Navbar.css` - Theme-aware styling
✅ **Updated:** `src/App.css` - Using CSS variables
✅ **Updated:** `src/index.css` - CSS variables definition

---

## Testing the Theme:

1. Start the app: `npm run dev`
2. Click the 🌙 (moon) icon in the navbar to enable dark mode
3. Click the ☀️ (sun) icon to switch back to light mode
4. Refresh the page - theme selection is saved!

---

## Best Practices:

✔️ Always use CSS variables instead of hardcoded colors
✔️ Test your components in both light and dark modes
✔️ Ensure sufficient contrast for accessibility
✔️ Use smooth transitions for better UX
✔️ Keep the theme switching instant (no delays)

Enjoy your themed app! 🎨
