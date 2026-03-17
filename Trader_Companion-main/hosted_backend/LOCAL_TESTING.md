# 🧪 Local Testing Guide

Test the hosted backend on your local machine before deploying!

## Quick Start (2 minutes)

### 1. Install Dependencies (if not already done)

```powershell
cd c:\Trader_Companion\hosted_backend
pip install -r requirements.txt
```

### 2. Start the Hosted Backend Server

```powershell
# From the hosted_backend directory
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
Price alert monitor started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

**The hosted backend is now running on `http://localhost:8001`** 🎉

> **Note**: Port 8001 is used to avoid conflict with Django backend on port 8000

### 3. Configure Your App

1. **Keep the hosted backend running** in this terminal
2. **Open your Trader Companion app** in the browser
3. Go to **Price Alerts** → **Hosted Backend** tab
4. Enter:
   - **Username**: `test` (or any name)
   - **Hosted URL**: `http://localhost:8001`
5. Click **Test** (should show green checkmark ✓)
6. Click **Save & Connect**

### 4. Test It Works

1. Create a test alert in the "Price Alerts" tab
2. Wait 30 seconds (auto-sync)
3. Go back to "Hosted Backend" tab
4. Click the refresh button (↻)
5. You should see your alert in the table!

## What's Happening?

```
Local Django Backend (Port 8000)
    ↓
Your Frontend (Port 5173)
    ↓ syncs to
Local Hosted Backend (Port 8001)
    ↓ monitors prices
    ↓ sends Telegram notifications
```

## Endpoints You Can Test

While the server is running, you can access:

- **Home**: http://localhost:8001
- **Health Check**: http://localhost:8001/health
- **API Docs**: http://localhost:8001/docs (Interactive Swagger UI!)
- **All Alerts**: http://localhost:8001/alerts

## Testing Features

### 1. View API Documentation
Open http://localhost:8000/docs in your browser to see all endpoints and test them interactively!

### 2. Manual API Testing

```powershell
# Check health
curl http://localhost:8001/health

# Get all alerts (will be empty initially)
curl http://localhost:8001/alerts

# After syncing from your app, check alerts for your user
curl http://localhost:8001/alerts/test
```

### 3. Test Telegram Notifications

In the app:
1. Make sure Telegram is configured (in "Enable Phone Notifications" tab)
2. Go to "Hosted Backend" tab
3. Click "Test" button
4. Check your Telegram! 📱

## Monitoring Logs

The terminal where you ran `python main.py` will show:
- Price check cycles every 30 seconds
- Alert triggers
- Telegram notification sends
- API requests from your frontend

Example log output:
```
Auto-synced alerts to hosted backend
Checking alert 1: AAPL @ $150.00
Current price: $149.50
Price alert monitor started
```

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Switching Between Local and Hosted

You can change the URL anytime!

**For Local Testing:**
- Hosted URL: `http://localhost:8001`

**For Production:**
- Hosted URL: `https://your-app.onrender.com`

Just update the URL in the "Hosted Backend" tab and click "Update & Reconnect"!

## Troubleshooting

### Port 8001 already in use

If you see `Error: Address already in use`:

**Option 1: Stop any process using port 8001**
```powershell
# Find what's using the port
netstat -ano | findstr :8001
# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F
```

**Option 2: Use a different port**
Edit `main.py`, change last line to:
```python
uvicorn.run(app, host="0.0.0.0", port=8002)
```
Then use `http://localhost:8002` in your app.

### Can't connect from frontend

Make sure:
- [ ] Hosted backend server is running (`python main.py`)
- [ ] No firewall blocking localhost
- [ ] URL is exactly `http://localhost:8001` (no trailing slash)
- [ ] Django backend is running on port 8000 (for local API calls)

### Alerts not syncing

Check:
- [ ] Django server is also running
- [ ] Frontend is running
- [ ] You clicked "Save & Connect"
- [ ] Wait 30 seconds for auto-sync

### Database not found

The database is created automatically on first run. If you see errors:
```powershell
# Delete the database and restart
Remove-Item alerts.db -ErrorAction SilentlyContinue
python main.py
```

## Development Tips

### View the Database

```powershell
# Install sqlite3 tools if needed
# Then:
sqlite3 alerts.db

# Inside sqlite:
.tables
SELECT * FROM alerts;
SELECT * FROM telegram_config;
.exit
```

### Check If It's Running

```powershell
# Quick health check
curl http://localhost:8001/health

# Should return: {"status":"healthy","timestamp":"..."}
```

### Hot Reload (Auto-restart on code changes)

Instead of `python main.py`, use:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Now the server auto-restarts when you edit `main.py`!

## Ready to Deploy?

Once local testing works, you're ready to deploy to a real hosting platform:
1. Follow `QUICK_START.md` for deployment
2. Update the Hosted URL in your app to the real URL
3. Everything keeps working, but now 24/7!

---

**Questions?** Check `HOSTED_BACKEND_SETUP.md` for deployment guides.
