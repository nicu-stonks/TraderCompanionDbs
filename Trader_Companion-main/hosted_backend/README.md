# Hosted Price Alerts Backend

A standalone FastAPI backend for hosting price alerts with automatic monitoring and Telegram notifications.

## Features

- Receives alert syncs from local backend
- Independently monitors stock prices using yfinance
- Sends Telegram notifications when alerts trigger
- Multi-user support with username-based authentication
- Automatic price checking every 30 seconds
- SQLite database for persistent storage

## Recommended Hosting Platforms (Free Tier)

### 1. **Render.com** (RECOMMENDED)
- **Free Tier**: Yes (750 hours/month)
- **Pros**: 
  - Easy deployment from GitHub
  - Automatic HTTPS
  - PostgreSQL/SQLite support
  - Auto-deploys on git push
  - Good for background services
- **Cons**: Spins down after 15 minutes of inactivity
- **Setup**: Connect GitHub repo, select Docker deployment
- **URL**: https://render.com

### 2. **Railway.app**
- **Free Tier**: $5 credit/month (usually enough for small apps)
- **Pros**:
  - Easy Docker deployment
  - Always-on services
  - Great developer experience
- **Cons**: Limited free credit
- **URL**: https://railway.app

### 3. **Fly.io**
- **Free Tier**: Yes (3 shared-cpu VMs)
- **Pros**:
  - Docker-native
  - Global deployment
  - Good performance
  - Always-on
- **Cons**: Requires credit card
- **URL**: https://fly.io

### 4. **Koyeb**
- **Free Tier**: Yes (1 app always-on)
- **Pros**:
  - Docker support
  - Always-on
  - Easy deployment
- **URL**: https://koyeb.com

## Deployment Instructions

### Option 1: Render.com (Easiest)

1. Push this folder to a GitHub repository
2. Go to https://render.com and sign up
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: price-alerts-backend
   - **Environment**: Docker
   - **Instance Type**: Free
6. Click "Create Web Service"
7. Copy the provided URL (e.g., `https://price-alerts-backend.onrender.com`)

### Option 2: Railway.app

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. From this directory: `railway init`
4. Deploy: `railway up`
5. Get URL: `railway domain`

### Option 3: Fly.io

1. Install Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
2. Login: `fly auth login`
3. From this directory: `fly launch`
4. Deploy: `fly deploy`
5. Get URL: `fly status`

## Local Testing

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

3. Test at: http://localhost:8000

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /alerts/sync` - Sync alerts from local backend
- `GET /alerts/{username}` - Get alerts for a user
- `GET /alerts` - Get all alerts from all users
- `POST /telegram/config` - Save Telegram configuration
- `GET /telegram/config/{username}` - Get Telegram configuration
- `POST /telegram/test` - Send test notification

## Environment Variables

None required for basic operation. Database is created automatically.

## Database

SQLite database (`alerts.db`) stores:
- User alerts
- Telegram configurations

Database is created automatically on first run.
