@echo off
setlocal

set "REPO_DIR=C:\Trader_Companion"

cd /d "%REPO_DIR%" || (
    echo Could not find %REPO_DIR%
    pause
    exit /b 1
)

echo Staging backup files in %REPO_DIR%...

REM 1. Clean up: Remove the dummy text file from Git and Disk
if exist last_run.txt del last_run.txt
git rm --cached last_run.txt 2>nul

REM 2. Force add the folders you specifically want (overrides .gitignore)
echo Force adding databases and media...
git add --force dbs
git add --force media_backup

REM 3. Add everything else normally
git add .

REM 4. Commit using --allow-empty (so we don't need the text file trick)
git commit --allow-empty -m "Auto backup: %date% %time%"

REM 5. Push
echo Pushing to GitHub...
git push origin main

echo.
echo Backup pushed to GitHub.
pause

endlocal