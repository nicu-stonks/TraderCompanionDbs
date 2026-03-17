#!/bin/bash

# Quick deployment script for hosted backend
# This script helps you deploy to various platforms quickly

echo "🚀 Hosted Backend Deployment Helper"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the hosted_backend directory"
    exit 1
fi

echo "Select deployment platform:"
echo "1) Render.com (Recommended - Free)"
echo "2) Railway.app (Free $5 credit)"
echo "3) Fly.io (Free tier)"
echo "4) Local testing"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "📦 Render.com Deployment"
        echo "========================"
        echo ""
        echo "Steps:"
        echo "1. Make sure you have Git initialized"
        echo "2. Push to GitHub"
        echo "3. Go to https://render.com"
        echo "4. Create New Web Service"
        echo "5. Connect your GitHub repo"
        echo "6. Select Docker environment"
        echo "7. Choose Free tier"
        echo ""
        
        read -p "Initialize Git repo? (y/n): " init_git
        if [ "$init_git" = "y" ]; then
            git init
            git add .
            git commit -m "Initial hosted backend for price alerts"
            echo "✅ Git initialized"
            echo ""
            echo "Now create a repo on GitHub and run:"
            echo "  git remote add origin https://github.com/YOUR_USERNAME/price-alerts-hosted.git"
            echo "  git push -u origin main"
        fi
        ;;
        
    2)
        echo ""
        echo "📦 Railway.app Deployment"
        echo "========================="
        echo ""
        
        # Check if railway CLI is installed
        if ! command -v railway &> /dev/null; then
            echo "Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        
        echo "Logging into Railway..."
        railway login
        
        echo "Initializing project..."
        railway init
        
        echo "Deploying..."
        railway up
        
        echo ""
        echo "Getting your URL..."
        railway domain
        
        echo ""
        echo "✅ Deployment complete!"
        ;;
        
    3)
        echo ""
        echo "📦 Fly.io Deployment"
        echo "===================="
        echo ""
        
        # Check if flyctl is installed
        if ! command -v flyctl &> /dev/null; then
            echo "❌ Fly CLI not installed"
            echo "Download from: https://fly.io/docs/hands-on/install-flyctl/"
            exit 1
        fi
        
        echo "Logging into Fly.io..."
        fly auth login
        
        echo "Launching app..."
        fly launch --name price-alerts-backend
        
        echo "Deploying..."
        fly deploy
        
        echo ""
        echo "Getting status..."
        fly status
        
        echo ""
        echo "✅ Deployment complete!"
        ;;
        
    4)
        echo ""
        echo "🧪 Local Testing"
        echo "================"
        echo ""
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python -m venv venv
        fi
        
        # Activate virtual environment
        echo "Activating virtual environment..."
        source venv/bin/activate
        
        # Install requirements
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        echo ""
        echo "Starting server..."
        echo "Server will be available at: http://localhost:8000"
        echo "API docs at: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop"
        echo ""
        
        python main.py
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Done! Copy the URL and paste it in your app's Hosted Backend tab"
