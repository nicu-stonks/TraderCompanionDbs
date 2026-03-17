@REM Create file C:\docker-secrets\ibeam.env with contents:
@REM IBEAM_ACCOUNT=username
@REM IBEAM_PASSWORD=password

@echo off
echo Starting Stocks Screener Services...

@echo off
REM Step 1: Backup files from app to outside

copy /Y "C:\Trader_Companion\Trader_Companion\dbs\personal_ranking.sqlite3" "C:\Trader_Companion\personal_ranking.sqlite3"
copy /Y "C:\Trader_Companion\Trader_Companion\dbs\trades_db.sqlite3" "C:\Trader_Companion\trades_db.sqlite3"
copy /Y "C:\Trader_Companion\Trader_Companion\dbs\db.sqlite3" "C:\Trader_Companion\db.sqlite3"
xcopy /E /I /Y "C:\Trader_Companion\Trader_Companion\media" "C:\Trader_Companion\media_backup"

REM Step 2: Update app from git
cd /d "C:\Trader_Companion\Trader_Companion"
git pull

REM Step 3: Restore files from outside to app

copy /Y "C:\Trader_Companion\personal_ranking.sqlite3" "C:\Trader_Companion\Trader_Companion\dbs\"
copy /Y "C:\Trader_Companion\trades_db.sqlite3" "C:\Trader_Companion\Trader_Companion\dbs\"
copy /Y "C:\Trader_Companion\db.sqlite3" "C:\Trader_Companion\Trader_Companion\dbs\"
xcopy /E /I /Y "C:\Trader_Companion\media_backup" "C:\Trader_Companion\Trader_Companion\media"


:: Start Django server
start cmd /k "cd C:\Trader_Companion\Trader_Companion && python .\manage.py runserver"

:: Start Flask microservice
start cmd /k "cd C:\Trader_Companion\Trader_Companion\flask_microservice_stocks_filterer && python .\api_endpoints.py"

:: Start React frontend
start cmd /k "cd C:\Trader_Companion\Trader_Companion\react-frontend-app && npm run dev"

:: Start Stock Buyer server
start cmd /k "cd C:\Trader_Companion\Trader_Companion\Buy_Seller\buy_seller_server && python .\stock_buyer.py"

:: Start Proxy Server
start cmd /k "cd C:\Trader_Companion\Trader_Companion\Buy_Seller\pivot_watchers && python .\proxy_server.py"

:: Start Ticker Data Fetcher server
start cmd /k "cd C:\Trader_Companion\Trader_Companion\Buy_Seller\ticker_data_fetcher && python .\server.py"

:: Start IBeam Docker container
@REM start cmd /k "docker run --restart always --env-file C:\docker-secrets\ibeam.env -p 5050:5000 voyz/ibeam"

echo All services started! You can close this window."C:\Users\hourt\Desktop\app.bat"