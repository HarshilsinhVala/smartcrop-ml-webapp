# Render Deployment Guide

## Step 1: Prepare Your Repository

1. **Commit all changes to GitHub:**
```bash
cd c:\mini_project-main
git add -A
git commit -m "Prepare for Render deployment"
git push origin main
```

## Step 2: Create Render Account & Connect GitHub

1. Go to [render.com](https://render.com)
2. Sign up or log in with your GitHub account
3. Click **"New +"** → **"Blueprint (YAML)"**
4. Connect your GitHub repository: `HarshilsinhVala/smartcrop-ml-webapp`
5. Select the **main** branch
6. Confirm the YAML file location: `/render.yaml`

## Step 3: Configure Environment Variables

Before deploying, set up these environment variables in Render:

### Backend Service (smartcrop-backend)
- `MONGODB_URI` - Your MongoDB connection string
- `NODE_ENV` - Set to `production`
- Add any other backend configs (JWT_SECRET, API_KEY, etc.)

### ML Service (smartcrop-ml)
- `FLASK_ENV` - Set to `production`
- `PORT` - Set to `5000`

### Frontend Service (smartcrop-frontend)
- `REACT_APP_API_URL` - Set to your backend URL (e.g., `https://smartcrop-backend.onrender.com`)
- `NODE_ENV` - Set to `production`

## Step 4: Deploy

1. In Render dashboard, click **"Create"**
2. It will detect the `render.yaml` file and create 3 services:
   - smartcrop-frontend (React on port 3000)
   - smartcrop-backend (Node.js on port 3001)
   - smartcrop-ml (Python on port 5000)

3. Wait for all services to build and deploy (this takes 5-15 minutes)

## Step 5: Get Your Live URLs

Once deployed, you'll get URLs like:
- Frontend: `https://smartcrop-frontend.onrender.com`
- Backend: `https://smartcrop-backend.onrender.com`
- ML: `https://smartcrop-ml.onrender.com`

## Important Notes

### ⚠️ Free Plan Limitations
- Services spin down after 15 minutes of inactivity (first request takes ~30 seconds)
- Limited to 400 compute hours/month
- 1 GB filesystem, 512 MB RAM

### ✅ Database Setup
Make sure your MongoDB URI is correctly set in environment variables.

### ✅ Auto-Sync with GitHub
Once you push to GitHub's `main` branch, Render will automatically redeploy your services!

### 🔗 Update API Calls
In your React code, update API calls to use the production backend URL:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'https://smartcrop-backend.onrender.com';
```

## Troubleshooting

### Service Won't Deploy
- Check build logs in Render dashboard
- Ensure `render.yaml` is in root directory
- Verify all environment variables are set

### API Calls Failing
- Check CORS settings in Backend (`server.js`)
- Verify `REACT_APP_API_URL` is set correctly
- Check Backend is actually running

### Python Dependencies Error
- Verify `requirements.txt` has all dependencies
- Check Python version compatibility (using 3.x)

## Manual Update Push (Optional)

If you need to manually trigger a redeployment:
```bash
git add -A
git commit -m "Update for production"
git push origin main
# Render will automatically redeploy!
```

---

**Need Help?**
- Render Docs: https://render.com/docs
- Check Render Dashboard Logs for detailed error messages
