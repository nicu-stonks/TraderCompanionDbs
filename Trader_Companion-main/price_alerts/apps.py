from django.apps import AppConfig
import logging
import sys

logger = logging.getLogger(__name__)


class PriceAlertsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'price_alerts'
    _monitor_started = False
    
    def ready(self):
        """Start the price alert monitor when Django is ready"""
        # Prevent starting during migrations or other management commands
        if 'migrate' in sys.argv or 'makemigrations' in sys.argv or 'test' in sys.argv:
            return
        
        # Prevent multiple starts (ready() can be called multiple times)
        if PriceAlertsConfig._monitor_started:
            return
        
        try:
            from .monitor import start_monitoring
            start_monitoring()
            PriceAlertsConfig._monitor_started = True
            print("=" * 60)
            print("PRICE ALERTS MONITOR: Started successfully")
            print("=" * 60)
            logger.info("Price alerts app initialized and monitoring started")
            logger.info("Monitor runs independently - alerts will work even if frontend is closed")
        except Exception as e:
            error_msg = f"Failed to start price alert monitor: {e}"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
