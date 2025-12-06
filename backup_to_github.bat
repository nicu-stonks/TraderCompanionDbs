@echo off
setlocal

set "REPO_DIR=C:\Trader_Companion"

cd /d "%REPO_DIR%" || (
    echo Could not find %REPO_DIR%
    pause
    exit /b 1
)

echo Staging backup files in %REPO_DIR%...

REM Optional: keep in sync if you ever pull from elsewhere
git pull --rebase

git add .
git commit -m "Auto backup %date% %time%" || echo Nothing to commit.

git push origin main

echo.
echo Backup pushed to GitHub.
pause

endlocal
