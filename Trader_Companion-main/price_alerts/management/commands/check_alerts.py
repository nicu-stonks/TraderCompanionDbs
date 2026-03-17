"""
Management command to check alert monitor status and manually start if needed.
Usage: python manage.py check_alerts
"""
from django.core.management.base import BaseCommand
from price_alerts.monitor import get_monitor, start_monitoring
from price_alerts.models import Alert


class Command(BaseCommand):
    help = 'Check price alert monitor status and start if needed'

    def handle(self, *args, **options):
        monitor = get_monitor()
        
        self.stdout.write("=" * 60)
        self.stdout.write("PRICE ALERTS MONITOR STATUS")
        self.stdout.write("=" * 60)
        
        if monitor.running:
            self.stdout.write(self.style.SUCCESS("Monitor is RUNNING"))
        else:
            self.stdout.write(self.style.WARNING("Monitor is NOT running"))
            self.stdout.write("Starting monitor...")
            start_monitoring()
            if monitor.running:
                self.stdout.write(self.style.SUCCESS("Monitor started successfully"))
            else:
                self.stdout.write(self.style.ERROR("Failed to start monitor"))
        
        # Show active alerts
        active_alerts = Alert.objects.filter(is_active=True, triggered=False)
        self.stdout.write(f"\nActive alerts: {active_alerts.count()}")
        
        if active_alerts.exists():
            self.stdout.write("\nActive alerts list:")
            for alert in active_alerts:
                current = f"${alert.current_price:.2f}" if alert.current_price else "N/A"
                self.stdout.write(f"  - {alert.ticker}: Alert @ ${alert.alert_price:.2f}, Current: {current}")
        else:
            self.stdout.write(self.style.WARNING("\nNo active alerts. Create alerts via the frontend or admin."))
        
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("Monitor runs independently - alerts work even if frontend is closed")
        self.stdout.write("=" * 60)

