"""
Database module for stock buyer trade storage.
Uses SQLite with thread-safe operations.
"""
import sqlite3
import json
import threading
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

# Database path relative to this file (../../dbs/stock_buyer.sqlite3)
DB_FILE = '../../dbs/stock_buyer.sqlite3'

def _get_db_path() -> str:
    """Get absolute path to the database, relative to this file's location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), DB_FILE))

# Thread-local storage for connections
_local = threading.local()

def get_connection() -> sqlite3.Connection:
    """Get thread-local database connection."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        db_path = _get_db_path()
        # Ensure dbs directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _local.connection = sqlite3.connect(db_path, check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row
    return _local.connection

@contextmanager
def get_cursor():
    """Context manager for database cursor with automatic commit/rollback."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def init_db():
    """Initialize database schema."""
    with get_cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_trades (
                trade_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                risk_amount REAL NOT NULL,
                lower_price_range REAL NOT NULL,
                higher_price_range REAL NOT NULL,
                order_type TEXT NOT NULL DEFAULT 'MKT',
                adaptive_priority TEXT,
                timeout_seconds INTEGER NOT NULL DEFAULT 5,
                sell_stops TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # Lightweight migration for existing databases.
        cursor.execute("PRAGMA table_info(pending_trades)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if 'order_type' not in existing_columns:
            cursor.execute("ALTER TABLE pending_trades ADD COLUMN order_type TEXT NOT NULL DEFAULT 'MKT'")
        if 'adaptive_priority' not in existing_columns:
            cursor.execute("ALTER TABLE pending_trades ADD COLUMN adaptive_priority TEXT")
        if 'timeout_seconds' not in existing_columns:
            cursor.execute("ALTER TABLE pending_trades ADD COLUMN timeout_seconds INTEGER NOT NULL DEFAULT 5")
        # Create index for faster lookups by criteria
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_trades_criteria 
            ON pending_trades(ticker, lower_price_range, higher_price_range)
        """)
        # Config table for persisting settings like available_risk
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

def save_risk_amount(amount: float) -> bool:
    """Save the available risk amount to database."""
    try:
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO config (key, value) VALUES ('available_risk', ?)
            """, (str(amount),))
            return True
    except Exception:
        return False

def load_risk_amount() -> float:
    """Load the available risk amount from database. Returns 0.0 if not set."""
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT value FROM config WHERE key = 'available_risk'")
            row = cursor.fetchone()
            if row:
                return float(row[0])
            return 0.0
    except Exception:
        return 0.0

def add_trade(trade_id: str, ticker: str, shares: float, risk_amount: float,
              lower_price_range: float, higher_price_range: float,
              sell_stops: List[Dict], order_type: str = 'MKT',
              adaptive_priority: Optional[str] = None, timeout_seconds: int = 5) -> bool:
    """Add a trade to pending_trades. Returns True if successful."""
    try:
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pending_trades 
                (trade_id, ticker, shares, risk_amount, lower_price_range, 
                 higher_price_range, order_type, adaptive_priority, timeout_seconds, sell_stops, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, ticker, shares, risk_amount, 
                lower_price_range, higher_price_range,
                order_type,
                adaptive_priority,
                int(timeout_seconds),
                json.dumps(sell_stops),
                datetime.now().isoformat()
            ))
            return cursor.rowcount == 1
    except sqlite3.IntegrityError:
        # Duplicate trade_id
        return False

def find_trade_by_criteria(ticker: str, lower_price: float, 
                           higher_price: float) -> Optional[Dict[str, Any]]:
    """Find a trade by ticker and price range."""
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT * FROM pending_trades 
            WHERE UPPER(ticker) = UPPER(?) 
            AND ABS(lower_price_range - ?) < 0.001
            AND ABS(higher_price_range - ?) < 0.001
        """, (ticker, lower_price, higher_price))
        row = cursor.fetchone()
        if row:
            return _row_to_dict(row)
        return None

def find_trade_by_id(trade_id: str) -> Optional[Dict[str, Any]]:
    """Find a trade by its ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM pending_trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_dict(row)
        return None

def delete_trade(trade_id: str) -> bool:
    """Delete a trade by ID. Returns True if a row was deleted."""
    with get_cursor() as cursor:
        cursor.execute("DELETE FROM pending_trades WHERE trade_id = ?", (trade_id,))
        return cursor.rowcount == 1

def trade_exists(trade_id: str) -> bool:
    """Check if a trade exists in pending_trades."""
    with get_cursor() as cursor:
        cursor.execute("SELECT 1 FROM pending_trades WHERE trade_id = ?", (trade_id,))
        return cursor.fetchone() is not None

def get_all_trades() -> List[Dict[str, Any]]:
    """Get all pending trades."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM pending_trades ORDER BY created_at")
        return [_row_to_dict(row) for row in cursor.fetchall()]

def get_trade_count() -> int:
    """Get count of pending trades."""
    with get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM pending_trades")
        return cursor.fetchone()[0]

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a database row to a dictionary with parsed sell_stops."""
    d = dict(row)
    d['sell_stops'] = json.loads(d['sell_stops'])
    return d
