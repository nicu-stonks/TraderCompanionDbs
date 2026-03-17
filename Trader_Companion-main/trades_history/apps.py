import sys
from django.apps import AppConfig


class TradesHistoryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trades_history"

    def ready(self):
        # Auto-run a script on startup to manually ensure the column exists in sqlite3, bypassing migration errors
        if 'runserver' in sys.argv:
            import threading
            
            def add_column():
                from django.db import connections
                try:
                    with connections['trades_db'].cursor() as cursor:
                        # Check if table exists and column is missing
                        cursor.execute("PRAGMA table_info(trades_history_trades)")
                        columns = [row[1] for row in cursor.fetchall()]
                        if columns and 'Pct_Of_Equity' not in columns:
                            print("Adding missing Pct_Of_Equity column to SQLite directly...")
                            cursor.execute("ALTER TABLE trades_history_trades ADD COLUMN Pct_Of_Equity real DEFAULT 0.0 NOT NULL")
                            print("Column Pct_Of_Equity added successfully.")
                except Exception as e:
                    print(f"Error checking/adding Pct_Of_Equity column: {e}")
            
            # Run in a thread to ensure Django is fully loaded before querying
            threading.Timer(2.0, add_column).start()
