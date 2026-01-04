@echo off
setlocal

set "REPO_DIR=C:\Trader_Companion"

cd /d "%REPO_DIR%" || (
    echo Could not find %REPO_DIR%
    pause
    exit /b 1
)

echo Staging backup files in %REPO_DIR%...

REM 1. Pull changes to prevent conflicts
git pull --rebase

REM 2. THE TRICK: Update a dummy file with the current time.
REM This forces a physical file change every single time.
echo Last backup run: %date% %time% > last_run.txt

REM 3. Add all files (including the updated last_run.txt)
git add .

REM 4. Commit (This will now ALWAYS succeed because last_run.txt changed)
git commit -m "Auto backup: %date% %time%"

REM 5. Push changes
echo Pushing to GitHub...
git push origin main

echo.
echo Backup pushed to GitHub.
pause

endlocal