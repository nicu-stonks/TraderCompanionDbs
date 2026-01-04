@echo off
setlocal

set "REPO_DIR=C:\Trader_Companion"

cd /d "%REPO_DIR%" || (
    echo Could not find %REPO_DIR%
    pause
    exit /b 1
)

echo Staging backup files in %REPO_DIR%...

REM Ensure we have the latest history before adding our new one
git pull --rebase

REM Add all files (if any changed)
git add .

REM --- THE FIX ---
REM --allow-empty forces a commit to be created even if files are identical.
REM This ensures there is always something new to push.
git commit --allow-empty -m "Force backup %date% %time%"

echo Pushing to GitHub...
git push origin main

echo.
echo Backup pushed to GitHub.
pause

endlocal