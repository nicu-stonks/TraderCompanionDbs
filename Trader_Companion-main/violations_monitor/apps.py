import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ViolationsMonitorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'violations_monitor'
    verbose_name = 'Violations Monitor'
    _compute_started = False

    def ready(self):
        """Start the background violations compute service when Django is ready."""
        if 'migrate' in sys.argv or 'makemigrations' in sys.argv or 'test' in sys.argv:
            return

        # Django runserver with autoreload calls ready() in a parent process and
        # again in the child process. Start compute service only in RUN_MAIN child
        # to avoid duplicate background loops and duplicated external HTTP traffic.
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            logger.debug('Skipping compute service startup in reloader parent process')
            return

        if ViolationsMonitorConfig._compute_started:
            return
        try:
            from .compute_service import start_compute_service
            start_compute_service()
            ViolationsMonitorConfig._compute_started = True
            logger.info('Violations compute service started — results update every 10s')
        except Exception as e:
            logger.error(f'Failed to start violations compute service: {e}')
