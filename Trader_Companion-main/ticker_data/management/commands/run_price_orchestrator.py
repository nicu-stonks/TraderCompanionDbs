from django.core.management.base import BaseCommand
import logging

from ticker_data.services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Runs the infinite background loop to fetch ticker data sequentially for rate-limiting."

    def handle(self, *args, **options):
        # Attach a handler only to the ticker_data logger so we don't capture
        # logs from other modules (price_alerts, violations_monitor, etc.) that
        # run as threads in the same Django process.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        ticker_data_logger = logging.getLogger('ticker_data')
        ticker_data_logger.setLevel(logging.INFO)
        ticker_data_logger.addHandler(handler)
        ticker_data_logger.propagate = False  # Don't let it bubble up to root
        
        self.stdout.write(self.style.SUCCESS("Starting Price Orchestrator loop..."))
        
        try:
            orchestrator = Orchestrator()
            orchestrator.run_loop()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\nLoop stopped gracefully via KeyboardInterrupt."))
