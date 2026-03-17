import sys
from django.apps import AppConfig
from django.core.management import call_command

class NutrientsTrackerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'nutrients_tracker'

    def ready(self):
        # Auto-run a script on startup to manually ensure the column exists in sqlite3, bypassing migration errors
        if 'runserver' in sys.argv:
            import threading
            
            def add_column():
                from django.db import connections
                try:
                    with connections['nutrients_tracker_db'].cursor() as cursor:
                        # Check if table exists and column is missing
                        cursor.execute("PRAGMA table_info(nutrients_tracker_dailyrecord)")
                        columns = [row[1] for row in cursor.fetchall()]
                        if columns and 'nutrient_sources' not in columns:
                            print("Adding missing nutrient_sources column to SQLite directly...")
                            cursor.execute("ALTER TABLE nutrients_tracker_dailyrecord ADD COLUMN nutrient_sources text DEFAULT '{}'")
                            print("Column added successfully.")
                except Exception as e:
                    print(f"Error checking/adding column: {e}")
            
            # Run in a thread to ensure Django is fully loaded before querying
            threading.Timer(2.0, add_column).start()
