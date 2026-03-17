import logging
from django.apps import AppConfig
from django.conf import settings
from django.db import connections
import os

logger = logging.getLogger(__name__)

class TickerDataConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ticker_data'

    def ready(self):
        self._ensure_sqlite_schema_raw_sql()

    def _ensure_sqlite_schema_raw_sql(self):
        """Create ticker_data sqlite file + tables/indexes via raw SQL when missing.

        This avoids migration dependency on fresh machines.
        """
        try:
            db_path = os.path.join(settings.BASE_DIR, 'dbs', 'ticker_data.sqlite3')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            if not os.path.exists(db_path):
                with open(db_path, 'a', encoding='utf-8'):
                    pass

            conn = connections['ticker_data_db']
            with conn.cursor() as cursor:
                # Improve SQLite concurrency under mixed read/write workload.
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_data_providersettings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        max_requests_per_10s REAL NOT NULL DEFAULT 10.0,
                        active_provider VARCHAR(20) NOT NULL DEFAULT 'yfinance',
                        updated_at DATETIME NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_data_trackedticker (
                        symbol VARCHAR(20) PRIMARY KEY,
                        added_at DATETIME NOT NULL,
                        last_trade_seen_at DATETIME NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_data_historicalprice (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        date DATE NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume BIGINT NOT NULL,
                        ingested_at DATETIME NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_data_historicalprice5m (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp DATETIME NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume BIGINT NOT NULL,
                        ingested_at DATETIME NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_data_historicalpriceweekly (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        date DATE NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume BIGINT NOT NULL,
                        ingested_at DATETIME NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_data_requestlog (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider VARCHAR(20) NOT NULL,
                        timestamp REAL NOT NULL,
                        symbol VARCHAR(20) NULL,
                        duration_ms REAL NULL,
                        success BOOLEAN NOT NULL DEFAULT 1
                    )
                """)

                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS ticker_data_historicalprice_symbol_date_uniq
                    ON ticker_data_historicalprice(symbol, date)
                """)
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS ticker_data_historicalprice5m_symbol_ts_uniq
                    ON ticker_data_historicalprice5m(symbol, timestamp)
                """)
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS ticker_data_historicalpriceweekly_symbol_date_uniq
                    ON ticker_data_historicalpriceweekly(symbol, date)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalprice_symbol_idx
                    ON ticker_data_historicalprice(symbol)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalprice_ingested_at_idx
                    ON ticker_data_historicalprice(ingested_at)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalprice5m_symbol_idx
                    ON ticker_data_historicalprice5m(symbol)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalprice5m_timestamp_idx
                    ON ticker_data_historicalprice5m(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalprice5m_ingested_at_idx
                    ON ticker_data_historicalprice5m(ingested_at)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalpriceweekly_symbol_idx
                    ON ticker_data_historicalpriceweekly(symbol)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalpriceweekly_date_idx
                    ON ticker_data_historicalpriceweekly(date)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_historicalpriceweekly_ingested_at_idx
                    ON ticker_data_historicalpriceweekly(ingested_at)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_requestlog_provider_idx
                    ON ticker_data_requestlog(provider)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_requestlog_timestamp_idx
                    ON ticker_data_requestlog(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS ticker_data_requestlog_symbol_idx
                    ON ticker_data_requestlog(symbol)
                """)

                # Idempotent column patching for existing DBs on other machines.
                self._ensure_column_exists(cursor, 'ticker_data_providersettings', 'max_requests_per_10s', 'REAL NOT NULL DEFAULT 10.0')
                self._ensure_column_exists(cursor, 'ticker_data_providersettings', 'active_provider', "VARCHAR(20) NOT NULL DEFAULT 'yfinance'")
                self._ensure_column_exists(cursor, 'ticker_data_providersettings', 'updated_at', 'DATETIME NOT NULL')

                self._ensure_column_exists(cursor, 'ticker_data_trackedticker', 'last_trade_seen_at', 'DATETIME NULL')

                self._ensure_column_exists(cursor, 'ticker_data_historicalprice', 'ingested_at', 'DATETIME NOT NULL')
                self._ensure_column_exists(cursor, 'ticker_data_historicalprice5m', 'ingested_at', 'DATETIME NOT NULL')
                self._ensure_column_exists(cursor, 'ticker_data_historicalpriceweekly', 'ingested_at', 'DATETIME NOT NULL')

                self._ensure_column_exists(cursor, 'ticker_data_requestlog', 'duration_ms', 'REAL NULL')
                self._ensure_column_exists(cursor, 'ticker_data_requestlog', 'success', 'BOOLEAN NOT NULL DEFAULT 1')
                self._ensure_column_exists(cursor, 'ticker_data_requestlog', 'request_count', 'INTEGER NOT NULL DEFAULT 1')

            logger.info("[TickerData] Raw SQL schema bootstrap completed for ticker_data_db")
        except Exception as exc:
            logger.error("[TickerData] Raw SQL schema bootstrap failed: %s", exc)

    def _ensure_column_exists(self, cursor, table_name: str, column_name: str, column_sql: str):
        """Add a missing SQLite column using ALTER TABLE in an idempotent way."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if column_name in existing_columns:
            return

        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
        )
        logger.info("[TickerData] Added missing column %s.%s", table_name, column_name)
