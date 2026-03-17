# Quick deployment script for Windows PowerShell
# Run this script to deploy hosted backend

Write-Host "Hosted Backend Deployment Helper" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if we are in the right directory
if (-not (Test-Path "main.py")) {
    Write-Host "Error: Please run this script from the hosted_backend directory" -ForegroundColor Red
    exit 1
}

Write-Host "Select deployment platform:"
Write-Host "1) Render.com (Recommended - Free)"
Write-Host "2) Railway.app (Free $5 credit)"
Write-Host "3) Fly.io (Free tier)"
Write-Host "4) Local testing"
Write-Host ""
$choice = Read-Host "Enter choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Render.com Deployment" -ForegroundColor Green
        Write-Host "========================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Steps:"
        Write-Host "1. Make sure you have Git initialized"
        Write-Host "2. Push to GitHub"
        Write-Host "3. Go to https://render.com"
        Write-Host "4. Create New Web Service"
        Write-Host "5. Connect your GitHub repo"
        Write-Host "6. Select Docker environment"
        Write-Host "7. Choose Free tier"
        Write-Host ""
        
        $init_git = Read-Host "Initialize Git repo? (y/n)"
        if ($init_git -eq "y") {
            git init
            git add .
            git commit -m "Initial hosted backend for price alerts"
            Write-Host "Git initialized" -ForegroundColor Green
            Write-Host ""
            Write-Host "Now create a repo on GitHub and run:"
            Write-Host "  git remote add origin https://github.com/YOUR_USERNAME/price-alerts-hosted.git"
            Write-Host "  git push -u origin main"
        }
    }
    
    "2" {
        Write-Host ""
        Write-Host "Railway.app Deployment" -ForegroundColor Green
        Write-Host "=========================" -ForegroundColor Green
        Write-Host ""
        
        # Check if railway CLI is installed
        $railwayExists = Get-Command railway -ErrorAction SilentlyContinue
        if (-not $railwayExists) {
            Write-Host "Installing Railway CLI..." -ForegroundColor Yellow
            npm install -g @railway/cli
        }
        
        Write-Host "Logging into Railway..." -ForegroundColor Yellow
        railway login
        
        Write-Host "Initializing project..." -ForegroundColor Yellow
        railway init
        
        Write-Host "Deploying..." -ForegroundColor Yellow
        railway up
        
        Write-Host ""
        Write-Host "Getting your URL..." -ForegroundColor Yellow
        railway domain
        
        Write-Host ""
        Write-Host "Deployment complete!" -ForegroundColor Green
    }
    
    "3" {
        Write-Host ""
        Write-Host "Fly.io Deployment" -ForegroundColor Green
        Write-Host "====================" -ForegroundColor Green
        Write-Host ""
        
        # Check if flyctl is installed
        $flyExists = Get-Command flyctl -ErrorAction SilentlyContinue
        if (-not $flyExists) {
            Write-Host "Fly CLI not installed" -ForegroundColor Red
            Write-Host "Download from: https://fly.io/docs/hands-on/install-flyctl/"
            exit 1
        }
        
        Write-Host "Logging into Fly.io..." -ForegroundColor Yellow
        fly auth login
        
        Write-Host "Launching app..." -ForegroundColor Yellow
        fly launch --name price-alerts-backend
        
        Write-Host "Deploying..." -ForegroundColor Yellow
        fly deploy
        
        Write-Host ""
        Write-Host "Getting status..." -ForegroundColor Yellow
        fly status
        
        Write-Host ""
        Write-Host "Deployment complete!" -ForegroundColor Green
    }
    
    "4" {
        Write-Host ""
        Write-Host "Local Testing" -ForegroundColor Green
        Write-Host "================" -ForegroundColor Green
        Write-Host ""
        
        # Check if virtual environment exists
        if (-not (Test-Path "venv")) {
            Write-Host "Creating virtual environment..." -ForegroundColor Yellow
            python -m venv venv
        }
        
        # Activate virtual environment
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & "venv\Scripts\Activate.ps1"
        
        # Install requirements
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        Write-Host ""
        Write-Host "Starting server..." -ForegroundColor Green
        Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "API docs at: http://localhost:8000/docs" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
        Write-Host ""
        
        python main.py
    }
    
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Done! Copy the URL and paste it in your app Hosted Backend tab" -ForegroundColor Green
