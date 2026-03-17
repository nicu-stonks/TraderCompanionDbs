# Hosted Backend Setup Guide

This guide will walk you through deploying and configuring the hosted backend for price alerts.

## 🎯 What You're Setting Up

A hosted backend that:
- Runs 24/7 on a free cloud platform
- Monitors your price alerts independently
- Sends Telegram notifications even when your laptop is off
- Automatically syncs with your local backend every 30 seconds

## 📋 Prerequisites

1. A GitHub account (for deployment)
2. A Telegram bot (set up in the "Enable Phone Notifications" tab first)
3. 10 minutes of your time

## 🚀 Step-by-Step Deployment

### Option 1: Render.com (Recommended - Easiest)

**Why Render?**
- Free tier: 750 hours/month (enough for 24/7 operation)
- Zero configuration needed
- Deploy directly from GitHub
- Automatic HTTPS

**Steps:**

1. **Create a GitHub Repository**
   ```bash
   # From the Trader_Companion folder
   cd hosted_backend
   git init
   git add .
   git commit -m "Initial hosted backend"
   
   # Create a new repo on GitHub and push
   git remote add origin https://github.com/YOUR_USERNAME/price-alerts-hosted.git
   git push -u origin main
   ```

2. **Deploy to Render**
   - Go to [render.com](https://render.com) and sign up (free)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub account
   - Select your newly created repository
   - Configure:
     - **Name**: `price-alerts-backend` (or any name you want)
     - **Environment**: Docker
     - **Branch**: main
     - **Instance Type**: Free
   - Click **"Create Web Service"**
   - Wait 5-10 minutes for deployment

3. **Get Your URL**
   - Once deployed, Render will show your URL: `https://price-alerts-backend-XXXX.onrender.com`
   - Copy this URL

4. **Configure in Your App**
   - Open your Trader Companion app
   - Go to **Price Alerts** → **Hosted Backend** tab
   - Enter:
     - **Username**: Choose any username (e.g., "mihai" or "laptop1")
     - **Hosted URL**: Paste the Render URL
   - Click **"Test"** to verify connection
   - Click **"Save & Connect"**
   - Done! ✅

### Option 2: Railway.app (Always-On, $5 Credit)

**Why Railway?**
- $5 free credit per month
- True always-on (doesn't sleep)
- Easy Docker deployment

**Steps:**

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy**
   ```bash
   cd c:\Trader_Companion\hosted_backend
   railway login
   railway init
   railway up
   railway domain
   ```

3. **Get Your URL**
   - The last command will show your URL
   - Copy it

4. **Configure in Your App** (same as Render above)

### Option 3: Fly.io (Advanced)

**Steps:**

1. **Install Fly CLI**
   - Download from: https://fly.io/docs/hands-on/install-flyctl/
   - For Windows: Use installer

2. **Deploy**
   ```bash
   cd c:\Trader_Companion\hosted_backend
   fly auth login
   fly launch --name price-alerts-backend
   fly deploy
   fly status
   ```

3. **Configure in Your App** (same as above)

## 🧪 Testing Your Setup

1. **In the Hosted Backend tab:**
   - Click **"Send Test Notification"**
   - You should receive a Telegram message within seconds

2. **Verify Auto-Sync:**
   - Create a new alert in the "Price Alerts" tab
   - Wait 30 seconds
   - Go to "Hosted Backend" tab
   - Click refresh icon (↻) next to "Hosted Backend Status"
   - Your new alert should appear in the table

## ⚙️ How It Works

### Auto-Sync Process
```
Local Backend (Your Laptop)
    ↓
    Syncs every 30 seconds
    ↓
Frontend (React App)
    ↓
    Sends alerts via HTTP
    ↓
Hosted Backend (Cloud)
    ↓
    Monitors prices every 30 seconds
    ↓
    Sends Telegram notifications
```

### When Your Laptop is Off
- ✅ Hosted backend continues running
- ✅ Monitors prices every 30 seconds
- ✅ Sends Telegram notifications
- ❌ Cannot sync new alerts (syncs resume when laptop is back on)

### When Your Laptop is On
- ✅ Auto-syncs alerts every 30 seconds
- ✅ Both local and hosted backends active
- ✅ Redundant notifications (you'll get them twice)

## 🔧 Configuration Options

### Username
- Simple identifier for your alerts
- No password needed (assumes all users are trusted)
- Use different usernames for multiple laptops/accounts
- Example: "mihai-laptop", "mihai-desktop", "trading-account-1"

### Multiple Users
The hosted backend supports multiple users:
- Each username maintains separate alerts
- All alerts are stored in the hosted backend
- You can see all users' alerts in the "Hosted Backend Status" section

## 📊 Monitoring

### View Hosted Alerts
In the "Hosted Backend" tab:
- **Total Alerts**: Number of all alerts across all users
- **Alerts by User**: Breakdown by username
- **Detailed Table**: All alerts with current prices and status

### Last Sync Time
Displayed at the bottom of the config card:
```
Last synced: 12/1/2025, 3:45:30 PM
```

## 🐛 Troubleshooting

### "Failed to connect to hosted backend"
- Check the URL is correct (include https://)
- Wait a few minutes after deployment (backend might still be starting)
- For Render.com: Service spins down after 15 min of inactivity, first request will be slow

### "Test notification failed"
- Ensure you configured Telegram in the "Enable Phone Notifications" tab
- Verify bot token and chat ID are correct
- Check hosted backend has been synced at least once

### Alerts not syncing
- Check your internet connection
- Verify laptop is on and app is running
- Check browser console for errors (F12 → Console tab)
- Try manual sync: Click "Sync Now" button

### Hosted backend stopped working
- For Render.com free tier: Service sleeps after 15 min inactivity
  - First alert check will wake it up (takes ~30 seconds)
  - To keep it always-on: Upgrade to paid tier ($7/month)

## 💡 Tips

1. **Keep it Running**
   - For Render.com: Service auto-wakes on first request
   - For Railway/Fly.io: Always-on by default

2. **Multiple Devices**
   - Use different usernames for each device
   - Each syncs independently
   - All alerts visible in hosted backend

3. **Battery Life**
   - Close laptop lid safely - hosted backend keeps running
   - No need to keep app open 24/7

4. **Security**
   - Only you can access your hosted backend
   - No sensitive data stored (except Telegram tokens)
   - Username-only auth (assumes trusted users)

## 🎉 You're Done!

Your price alerts will now continue working even when your laptop is off. 

Test it:
1. Create an alert
2. Wait for it to sync (30 seconds)
3. Close your laptop
4. Alert will still trigger via Telegram! 📱

## 📝 Quick Reference

### Render.com
- URL: https://render.com
- Free Tier: ✅ 750 hours/month
- Always-On: ❌ (sleeps after 15 min)
- Setup: Easiest

### Railway.app
- URL: https://railway.app
- Free Tier: ✅ $5 credit/month
- Always-On: ✅
- Setup: Easy

### Fly.io
- URL: https://fly.io
- Free Tier: ✅ 3 VMs
- Always-On: ✅
- Setup: Medium

---

Need help? Check the hosted backend logs:
- **Render**: Dashboard → Logs tab
- **Railway**: Dashboard → Deployments → Logs
- **Fly.io**: `fly logs`
