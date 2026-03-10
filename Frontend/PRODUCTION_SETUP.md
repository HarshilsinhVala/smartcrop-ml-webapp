# Production Setup - Run App Continuously

This guide explains how to run your Frontend React app continuously in production.

## Quick Start (Windows)

### Option 1: Using PM2 (Recommended - Auto-restart on crash)

#### 1. Install PM2 Globally
```bash
npm install -g pm2
```

#### 2. Build & Start with PM2
```bash
npm run prod
```

Then stop the running server and start it with PM2:
```bash
# Stop current server (Ctrl+C in terminal)

# Start with PM2
pm2 start server.js --name "mini-project-frontend"

# Enable auto-start on Windows boot
pm2 save
pm2 startup windows
```

#### 3. Monitor the App
```bash
pm2 status                    # View running apps
pm2 logs mini-project-frontend  # View live logs
pm2 reload mini-project-frontend  # Graceful restart
pm2 stop mini-project-frontend    # Stop app
pm2 restart mini-project-frontend # Restart app
```

---

### Option 2: Using Windows Task Scheduler (Auto-start on Boot)

#### 1. Build the React App
```bash
npm run build
```

#### 2. Create a Batch File

Create `start_app.bat` in the Frontend directory:
```batch
@echo off
cd /d C:\mini_project-main\Frontend
node server.js
pause
```

#### 3. Schedule with Task Scheduler
- Open **Task Scheduler** (Windows)
- Click **Create Basic Task**
- **Name**: "Mini Project Frontend"
- **Trigger**: "At startup"
- **Action**: 
  - Program: `C:\mini_project-main\Frontend\start_app.bat`
  - Start in: `C:\mini_project-main\Frontend`

---

### Option 3: Using Node.js Service (Advanced)

Install NSSM (Non-Sucking Service Manager):
```bash
# Download from https://nssm.cc/
# Or use chocolatey:
choco install nssm

# Create service
nssm install MiniProjectFrontend C:\mini_project-main\Frontend\server.js
nssm start MiniProjectFrontend
```

---

## Before Running in Production

1. **Build React App** (creates optimized `build/` folder)
   ```bash
   npm run build
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Set Environment Variables** (if needed)
   Create `.env` in Frontend directory:
   ```
   PORT=5000
   REACT_APP_API_URL=http://localhost:5000
   ```

4. **Test Locally**
   ```bash
   npm run prod
   ```
   Open http://localhost:5000 in browser

---

## File Structure After Build

```
Frontend/
├── build/                 # Production build (created by npm run build)
│   ├── index.html
│   ├── static/
│   └── ...
├── node_modules/
├── public/
├── src/
├── server.js              # Express server (serves React + API)
├── package.json
└── PRODUCTION_SETUP.md    # This file
```

---

## Troubleshooting

### Port Already in Use
```bash
# Change port in server.js or via environment variable
set PORT=3000
npm run serve
```

### App Crashes
- Check logs: `pm2 logs mini-project-frontend`
- Ensure Python backend is running (for predictions)
- Check database connection

### Files Not Serving
- Verify `build/` folder exists: `npm run build`
- Check `server.js` has static middleware configured
- Clear browser cache (Ctrl+Shift+Delete)

---

## Monitoring & Maintenance

### Check if App is Running
```bash
pm2 status
```

### View Real-time Logs
```bash
pm2 logs -n 100 mini-project-frontend
```

### Auto-restart Failed App
PM2 automatically restarts the app if it crashes.

### Stop/Restart
```bash
pm2 stop mini-project-frontend      # Graceful stop
pm2 restart mini-project-frontend   # Restart
pm2 delete mini-project-frontend    # Remove from PM2
```

---

## Server Configuration

The `server.js` file now:
- ✅ Serves React build files as static content
- ✅ Handles API endpoints for uploads & predictions
- ✅ Routes client-side navigation to React
- ✅ Supports environment variable for PORT

---

## Next Steps

1. Build the app: `npm run build`
2. Test locally: `npm run prod`
3. Deploy to production server
4. Use PM2 to keep it running

For questions or issues, check the logs with `pm2 logs`
