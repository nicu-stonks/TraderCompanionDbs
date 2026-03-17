"""
Standalone FastAPI backend for hosting price alerts
This app receives alerts from local backend and monitors them independently
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta, time as dt_time
import sqlite3
import yfinance as yf
import asyncio
import aiohttp
import os
import time
import pytz
from contextlib import asynccontextmanager

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'alerts.db')

# Global task handle
monitor_task = None
price_task = None

# Price cache and ticker scheduling
price_cache: Dict[str, Dict[str, object]] = {}
active_tickers: List[str] = []
ticker_refresh_interval = 300  # seconds between active ticker refreshes
last_ticker_refresh = 0.0
ticker_round_robin_index = 0

# Market schedule (Eastern Time)
market_open_hour = 9
market_open_minute = 30
market_close_hour = 16
market_close_minute = 0
market_check_interval = 30  # seconds between closed-market checks
price_request_interval = 0.5  # seconds between ticker fetches
price_max_age = timedelta(minutes=5)


class Alert(BaseModel):
    """Alert model"""
    id: Optional[int] = None
    username: str
    ticker: str
    alert_price: float
    is_active: bool = True
    triggered: bool = False
    created_at: Optional[str] = None
    triggered_at: Optional[str] = None
    current_price: Optional[float] = None
    last_checked: Optional[str] = None
    initial_price_above_alert: Optional[bool] = None


class TelegramConfig(BaseModel):
    """Telegram configuration model"""
    username: str
    bot_token: str
    chat_id: str
    enabled: bool = True


class TestNotification(BaseModel):
    """Test notification request"""
    username: str


class AlertsSyncPayload(BaseModel):
    """Payload for syncing alerts and optional telegram config"""
    alerts: List[Alert]
    telegram_config: Optional[TelegramConfig] = None


def get_current_time_et() -> datetime:
    """Return current time in US/Eastern timezone."""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)


def get_time_until_next_open(current_time: datetime) -> timedelta:
    """Compute timedelta until next market open from current Eastern time."""
    next_open = current_time.replace(
        hour=market_open_hour,
        minute=market_open_minute,
        second=0,
        microsecond=0
    )

    if current_time >= next_open:
        next_open += timedelta(days=1)

    while next_open.weekday() >= 5:  # Skip weekends
        next_open += timedelta(days=1)

    return next_open - current_time


def is_market_open() -> bool:
    """Determine if US markets are open."""
    now = get_current_time_et()
    if now.weekday() >= 5:
        return False

    open_time = dt_time(market_open_hour, market_open_minute)
    close_time = dt_time(market_close_hour, market_close_minute)
    current_time = now.time()
    return open_time <= current_time <= close_time


def init_db():
    """Initialize the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            ticker TEXT NOT NULL,
            alert_price REAL NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            triggered BOOLEAN DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            triggered_at TEXT,
            current_price REAL,
            last_checked TEXT,
            initial_price_above_alert BOOLEAN
        )
    ''')
    
    # Create telegram_config table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telegram_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            bot_token TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            enabled BOOLEAN DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


def refresh_active_tickers(force: bool = False):
    """Refresh the cached list of unique tickers with active alerts."""
    global active_tickers, last_ticker_refresh, ticker_round_robin_index

    now = time.time()
    if not force and now - last_ticker_refresh < ticker_refresh_interval:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT UPPER(ticker)
        FROM alerts
        WHERE is_active = 1
    ''')
    rows = cursor.fetchall()
    conn.close()

    tickers = [row[0] for row in rows if row[0]]
    active_tickers = tickers
    ticker_round_robin_index = 0 if tickers else 0
    last_ticker_refresh = now


def upsert_telegram_config(cursor: sqlite3.Cursor, config: TelegramConfig):
    """Insert or update telegram configuration for a user."""
    cursor.execute('''
        INSERT INTO telegram_config (username, bot_token, chat_id, enabled, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            bot_token = excluded.bot_token,
            chat_id = excluded.chat_id,
            enabled = excluded.enabled,
            updated_at = excluded.updated_at
    ''', (config.username, config.bot_token, config.chat_id, config.enabled, datetime.now().isoformat()))


async def prime_initial_prices_for_user(username: str, tickers: List[str]):
    """Fetch initial prices for synced tickers so hosted backend owns state."""
    unique_tickers = sorted({(ticker or '').strip().upper() for ticker in tickers if ticker})
    if not unique_tickers:
        return

    updates = []
    for ticker in unique_tickers:
        price = await fetch_price_for_ticker(ticker)
        if price is None:
            continue

        now_dt = datetime.now()
        price_cache[ticker] = {
            "price": price,
            "updated_at": now_dt
        }
        updates.append((price, now_dt.isoformat(), price, ticker))

    if not updates:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        for current_price, timestamp_iso, comparison_price, ticker in updates:
            cursor.execute('''
                UPDATE alerts
                SET current_price = ?,
                    last_checked = ?,
                    initial_price_above_alert = CASE
                        WHEN initial_price_above_alert IS NULL THEN
                            CASE WHEN ? > alert_price THEN 1 ELSE 0 END
                        ELSE initial_price_above_alert
                    END
                WHERE username = ? AND UPPER(ticker) = ?
            ''', (current_price, timestamp_iso, comparison_price, username, ticker))

        conn.commit()
    finally:
        conn.close()


async def fetch_price_for_ticker(ticker: str) -> Optional[float]:
    """Fetch the latest price for a ticker using yfinance."""

    def _fetch() -> Optional[float]:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            return float(price) if price is not None else None
        except Exception as exc:
            print(f"Error fetching price for {ticker}: {exc}")
            return None

    return await asyncio.to_thread(_fetch)


async def ticker_price_loop():
    """Continuously refresh ticker prices in a round-robin fashion."""
    global ticker_round_robin_index

    refresh_active_tickers(force=True)

    while True:
        if not active_tickers:
            refresh_active_tickers(force=True)
            await asyncio.sleep(5)
            continue

        refresh_active_tickers()

        if not active_tickers:
            await asyncio.sleep(5)
            continue

        if not is_market_open():
            await asyncio.sleep(market_check_interval)
            continue

        ticker = active_tickers[ticker_round_robin_index]
        ticker_round_robin_index = (ticker_round_robin_index + 1) % len(active_tickers)

        price = await fetch_price_for_ticker(ticker)
        if price is not None:
            price_cache[ticker] = {
                "price": price,
                "updated_at": datetime.now()
            }

        await asyncio.sleep(price_request_interval)


async def send_telegram_notification(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram notification"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return True
                else:
                    print(f"Telegram API error: {response.status}")
                    return False
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")
        return False


async def check_alerts():
    """Background task to check all active alerts"""
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get all active, non-triggered alerts
            cursor.execute('''
                SELECT id, username, ticker, alert_price, current_price, initial_price_above_alert
                FROM alerts
                WHERE is_active = 1 AND triggered = 0
            ''')
            
            alerts = cursor.fetchall()
            
            # Process each alert
            for alert_id, username, ticker, alert_price, current_price, initial_price_above in alerts:
                try:
                    ticker_key = ticker.upper()
                    cached_entry = price_cache.get(ticker_key)
                    latest_price = None

                    if cached_entry:
                        updated_at = cached_entry.get("updated_at")
                        if isinstance(updated_at, datetime):
                            if datetime.now() - updated_at <= price_max_age:
                                latest_price = cached_entry.get("price")

                    if latest_price is None:
                        latest_price = await fetch_price_for_ticker(ticker_key)
                        if latest_price is not None:
                            price_cache[ticker_key] = {
                                "price": latest_price,
                                "updated_at": datetime.now()
                            }

                    if latest_price is None:
                        continue

                    now_iso = datetime.now().isoformat()

                    if initial_price_above is None:
                        initial_price_above = 1 if latest_price > alert_price else 0
                        cursor.execute('''
                            UPDATE alerts 
                            SET initial_price_above_alert = ?, current_price = ?, last_checked = ?
                            WHERE id = ?
                        ''', (initial_price_above, latest_price, now_iso, alert_id))
                    else:
                        cursor.execute('''
                            UPDATE alerts 
                            SET current_price = ?, last_checked = ?
                            WHERE id = ?
                        ''', (latest_price, now_iso, alert_id))

                    should_trigger = False
                    if initial_price_above == 1:
                        should_trigger = latest_price <= alert_price
                    else:
                        should_trigger = latest_price >= alert_price

                    if should_trigger:
                        cursor.execute('''
                            UPDATE alerts 
                            SET triggered = 1, triggered_at = ?
                            WHERE id = ?
                        ''', (now_iso, alert_id))

                        cursor.execute('''
                            SELECT bot_token, chat_id, enabled
                            FROM telegram_config
                            WHERE username = ?
                        ''', (username,))

                        telegram_row = cursor.fetchone()
                        if telegram_row and telegram_row[2]:
                            bot_token, chat_id, _ = telegram_row
                            direction = "dropped to" if initial_price_above else "rose to"
                            message = (
                                "🚨 <b>PRICE ALERT TRIGGERED</b> 🚨\n\n"
                                f"Ticker: <b>{ticker}</b>\n"
                                f"Alert Price: <b>${alert_price:.2f}</b>\n"
                                f"Current Price: <b>${latest_price:.2f}</b>\n"
                                f"Price {direction} alert level!"
                            )

                            await send_telegram_notification(bot_token, chat_id, message)

                except Exception as e:
                    print(f"Error checking alert {alert_id}: {e}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error in check_alerts loop: {e}")
        
        # Wait 30 seconds before next check
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global monitor_task, price_task
    
    # Startup
    init_db()
    refresh_active_tickers(force=True)
    monitor_task = asyncio.create_task(check_alerts())
    price_task = asyncio.create_task(ticker_price_loop())
    print("Price alert monitor started")
    
    yield
    
    # Shutdown
    for task in (monitor_task, price_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    print("Price alert monitor stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Hosted Price Alerts Backend",
    description="Standalone backend for price alerts with Telegram notifications",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hosted Price Alerts Backend",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/alerts/sync")
async def sync_alerts(payload: AlertsSyncPayload):
    """Sync alerts (and optional telegram config) from local backend"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    username: Optional[str] = None
    tickers_for_prime: List[str] = []
    result_payload = {"message": "No alerts to sync", "synced": 0}

    try:
        alerts = payload.alerts or []
        telegram_config = payload.telegram_config

        if not alerts:
            if telegram_config:
                upsert_telegram_config(cursor, telegram_config)
                conn.commit()
                return {"message": "Telegram config updated", "synced": 0}
            return {"message": "No alerts to sync", "synced": 0}
        
        username = alerts[0].username
        
        # Load existing alerts for the user to diff against incoming payload
        cursor.execute('''
            SELECT id, ticker, alert_price, is_active, triggered,
                   created_at, triggered_at
            FROM alerts
            WHERE username = ?
        ''', (username,))
        existing_rows = cursor.fetchall()
        existing_by_id = {row[0]: row for row in existing_rows}
        existing_by_key: Dict[tuple, List[int]] = {}
        for row in existing_rows:
            key = (
                (row[1] or '').upper(),
                float(row[2]) if row[2] is not None else None,
                bool(row[3]),
                bool(row[4]),
                row[5]
            )
            existing_by_key.setdefault(key, []).append(row[0])

        existing_ids = {row[0] for row in existing_rows}
        ids_to_keep = set()
        inserted = 0
        updated = 0
        tickers_for_prime = []

        # Upsert alerts; hosted backend still derives current prices itself
        for alert in alerts:
            matched_id = None
            if alert.id is not None and alert.id in existing_by_id:
                matched_id = alert.id
            else:
                lookup_key = (
                    (alert.ticker or '').strip().upper(),
                    alert.alert_price,
                    bool(alert.is_active),
                    bool(alert.triggered),
                    alert.created_at
                )
                candidate_ids = existing_by_key.get(lookup_key)
                if candidate_ids:
                    matched_id = candidate_ids.pop(0)
                    if not candidate_ids:
                        existing_by_key.pop(lookup_key, None)
            if matched_id is None:
                cursor.execute('''
                    INSERT INTO alerts (
                        username, ticker, alert_price, is_active, triggered,
                        created_at, triggered_at, current_price, last_checked,
                        initial_price_above_alert
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.username, alert.ticker, alert.alert_price,
                    alert.is_active, alert.triggered, alert.created_at,
                    alert.triggered_at, None, None, None
                ))

                new_id = cursor.lastrowid
                ids_to_keep.add(new_id)
                inserted += 1
                if alert.ticker:
                    tickers_for_prime.append(alert.ticker)
            else:
                cursor.execute('''
                    UPDATE alerts SET
                        ticker = ?,
                        alert_price = ?,
                        is_active = ?,
                        triggered = ?,
                        created_at = ?,
                        triggered_at = ?
                    WHERE id = ?
                ''', (
                    alert.ticker, alert.alert_price, alert.is_active,
                    alert.triggered, alert.created_at, alert.triggered_at,
                    matched_id
                ))

                ids_to_keep.add(matched_id)
                updated += 1
        
        if telegram_config:
            upsert_telegram_config(cursor, telegram_config)
        
        # Remove alerts that no longer exist locally
        ids_to_delete = existing_ids - ids_to_keep
        if ids_to_delete:
            cursor.executemany('DELETE FROM alerts WHERE id = ?', ((alert_id,) for alert_id in ids_to_delete))

        conn.commit()
        refresh_active_tickers(force=True)
        result_payload = {
            "message": f"Synced {len(alerts)} alerts for user {username}",
            "synced": len(alerts),
            "inserted": inserted,
            "updated": updated,
            "deleted": len(ids_to_delete)
        }
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    if username and tickers_for_prime:
        await prime_initial_prices_for_user(username, tickers_for_prime)
    
    return result_payload


@app.get("/alerts/{username}")
async def get_alerts(username: str):
    """Get all alerts for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, ticker, alert_price, is_active, triggered,
               created_at, triggered_at, current_price, last_checked,
               initial_price_above_alert
        FROM alerts
        WHERE username = ?
        ORDER BY created_at DESC
    ''', (username,))
    
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alerts.append({
            "id": row[0],
            "username": row[1],
            "ticker": row[2],
            "alert_price": row[3],
            "is_active": bool(row[4]),
            "triggered": bool(row[5]),
            "created_at": row[6],
            "triggered_at": row[7],
            "current_price": row[8],
            "last_checked": row[9],
            "initial_price_above_alert": bool(row[10]) if row[10] is not None else None
        })
    
    return {"username": username, "alerts": alerts, "total": len(alerts)}


@app.get("/alerts")
async def get_all_alerts():
    """Get all alerts from all users"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get total count by user
    cursor.execute('''
        SELECT username, COUNT(*) as count
        FROM alerts
        GROUP BY username
    ''')
    user_counts = cursor.fetchall()
    
    # Get all alerts
    cursor.execute('''
        SELECT id, username, ticker, alert_price, is_active, triggered,
               created_at, triggered_at, current_price, last_checked,
               initial_price_above_alert
        FROM alerts
        ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alerts.append({
            "id": row[0],
            "username": row[1],
            "ticker": row[2],
            "alert_price": row[3],
            "is_active": bool(row[4]),
            "triggered": bool(row[5]),
            "created_at": row[6],
            "triggered_at": row[7],
            "current_price": row[8],
            "last_checked": row[9],
            "initial_price_above_alert": bool(row[10]) if row[10] is not None else None
        })
    
    user_stats = {username: count for username, count in user_counts}
    
    return {
        "alerts": alerts,
        "total": len(alerts),
        "by_user": user_stats
    }


@app.post("/telegram/config")
async def save_telegram_config(config: TelegramConfig):
    """Save or update Telegram configuration for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        upsert_telegram_config(cursor, config)
        conn.commit()
        return {"message": "Telegram config saved successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/telegram/config/{username}")
async def get_telegram_config(username: str):
    """Get Telegram configuration for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT username, bot_token, chat_id, enabled, created_at, updated_at
        FROM telegram_config
        WHERE username = ?
    ''', (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Telegram config not found")
    
    return {
        "username": row[0],
        "bot_token": row[1],
        "chat_id": row[2],
        "enabled": bool(row[3]),
        "created_at": row[4],
        "updated_at": row[5]
    }


@app.post("/telegram/test")
async def test_telegram_notification(test_request: TestNotification):
    """Send a test Telegram notification"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT bot_token, chat_id, enabled
        FROM telegram_config
        WHERE username = ?
    ''', (test_request.username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Telegram config not found")
    
    bot_token, chat_id, enabled = row
    
    if not enabled:
        raise HTTPException(status_code=400, detail="Telegram notifications are disabled")
    
    message = f"🧪 <b>TEST NOTIFICATION</b>\n\nThis is a test notification from the hosted backend for user: <b>{test_request.username}</b>\n\nIf you received this, your Telegram integration is working! ✅"
    
    success = await send_telegram_notification(bot_token, chat_id, message)
    
    if success:
        return {"message": "Test notification sent successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send test notification")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
