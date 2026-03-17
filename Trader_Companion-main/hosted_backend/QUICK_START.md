# 🚀 Quick Start Checklist

Follow this checklist to get your hosted backend running in under 10 minutes!

## ✅ Pre-Deployment Checklist

- [ ] Telegram bot configured (in "Enable Phone Notifications" tab)
- [ ] GitHub account created
- [ ] Internet connection available

## 📦 Step 1: Deploy Hosted Backend (5 minutes)

Choose ONE platform:

### Option A: Render.com (Recommended)
- [ ] Go to https://render.com and sign up
- [ ] Create a new GitHub repository
- [ ] Copy `hosted_backend/` folder to the repo
- [ ] Push to GitHub:
  ```bash
  cd hosted_backend
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin YOUR_REPO_URL
  git push -u origin main
  ```
- [ ] On Render: New Web Service → Connect GitHub → Select Docker
- [ ] Wait for deployment (5-10 minutes)
- [ ] Copy the URL (e.g., `https://your-app.onrender.com`)

### Option B: Railway.app (Faster)
- [ ] Install Railway CLI: `npm install -g @railway/cli`
- [ ] Run deployment script:
  ```bash
  cd c:\Trader_Companion\hosted_backend
  .\deploy.ps1
  ```
- [ ] Choose option 2 (Railway)
- [ ] Follow prompts
- [ ] Copy the generated URL

### Option C: Fly.io
- [ ] Install Fly CLI from https://fly.io/docs/hands-on/install-flyctl/
- [ ] Run:
  ```bash
  cd c:\Trader_Companion\hosted_backend
  fly launch
  fly deploy
  ```
- [ ] Copy the generated URL

## 🔧 Step 2: Configure in Your App (2 minutes)

- [ ] Open Trader Companion app
- [ ] Go to **Price Alerts** tab
- [ ] Click **"Hosted Backend"** tab (new!)
- [ ] Enter:
  - Username: `_________` (e.g., "mihai")
  - Hosted URL: `_________` (paste from Step 1)
- [ ] Click **"Test"** button
- [ ] Wait for green checkmark ✅
- [ ] Click **"Save & Connect"**

## 🧪 Step 3: Test It Works (3 minutes)

- [ ] In Hosted Backend tab, click **"Send Test Notification"**
- [ ] Check Telegram - you should receive a test message 📱
- [ ] Go to **"Price Alerts"** tab
- [ ] Create a test alert (any ticker, any price)
- [ ] Wait 30 seconds
- [ ] Go back to **"Hosted Backend"** tab
- [ ] Click refresh icon (↻) next to "Hosted Backend Status"
- [ ] Verify your alert appears in the table
- [ ] Delete the test alert

## ✨ Step 4: Verify Auto-Sync (Optional)

- [ ] Create a new alert
- [ ] Open browser Developer Tools (F12)
- [ ] Go to Console tab
- [ ] Look for: "Auto-synced alerts to hosted backend"
- [ ] Message should appear every 30 seconds

## 🎉 You're Done!

### What works now:
✅ Alerts sync automatically every 30 seconds  
✅ Hosted backend monitors prices 24/7  
✅ Telegram notifications work even when laptop is off  
✅ No manual intervention needed  

### Try this:
1. Create an alert below current price
2. Close your laptop
3. Wait for price to hit
4. Receive Telegram notification! 📱

---

## 🐛 Troubleshooting

### Test button shows "Connection failed"
- [ ] Check URL has `https://` prefix
- [ ] Wait 2-3 minutes after deployment
- [ ] For Render: First request can be slow (service waking up)
- [ ] Try pasting URL in browser - should see: `{"message": "Hosted Price Alerts Backend"}`

### Test notification fails
- [ ] Go to "Enable Phone Notifications" tab
- [ ] Verify bot token and chat ID are filled
- [ ] Click "Test Notification" there first
- [ ] If that works, return to Hosted Backend and try again

### Alerts not showing in hosted backend
- [ ] Check "Last synced" time at bottom
- [ ] Click "Sync Now" button
- [ ] Wait 30 seconds
- [ ] Click refresh icon (↻)

### Nothing works
- [ ] Check internet connection
- [ ] Restart the app
- [ ] Check browser console (F12 → Console) for errors
- [ ] Verify hosted backend is running (paste URL in browser)

---

## 📝 Notes

- **First sync**: Happens immediately after clicking "Save & Connect"
- **Auto-sync interval**: 30 seconds (configurable in code)
- **Price check interval**: 30 seconds on hosted backend
- **Render sleep**: Free tier sleeps after 15 min inactivity (auto-wakes on first request)

## 🆘 Need Help?

Check these files:
- `HOSTED_BACKEND_SETUP.md` - Detailed setup guide
- `hosted_backend/README.md` - Platform-specific instructions
- `IMPLEMENTATION_SUMMARY.md` - Technical overview

---

**Last Updated**: December 2025  
**Estimated Setup Time**: 10 minutes  
**Difficulty**: Easy 🟢
