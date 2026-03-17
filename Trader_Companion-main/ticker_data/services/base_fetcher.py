import time
import logging
from typing import Optional
from django.utils import timezone
from ..models import ProviderSettings, RequestLog, HistoricalPrice, HistoricalPriceWeekly

logger = logging.getLogger(__name__)

class BaseFetcher:
    PROVIDER_NAME = "base"

    def __init__(self):
        self._window_start = time.time()
        self._requests_in_window = 0
        self._last_fetch_end: Optional[float] = None
    
    def _enforce_rate_limit(self, consumed_requests: int = 2):
        """
        Implements a pacing (leaky bucket) algorithm that strictly tracks expected
        run times to guarantee evenly spaced requests over a 10s window.
        """
        settings = ProviderSettings.load()
        limit = settings.max_requests_per_10s
        
        if limit <= 0:
            return  # Rate limiting disabled
            
        period_seconds = 10.0
        ideal_interval = period_seconds / limit
        
        now = time.time()
        elapsed = now - self._window_start
        
        # If we exceeded the 10s window entirely, start a new window
        if elapsed >= period_seconds:
            self._window_start = now
            self._requests_in_window = 0
            elapsed = 0.0
            
        # Expected time for the CURRENT request
        expected_time = self._requests_in_window * ideal_interval
        
        if elapsed < expected_time:
            # We are ahead of schedule. Wait to match the exact pace.
            time.sleep(expected_time - elapsed)
            
        # Increment request count for this window
        self._requests_in_window += consumed_requests

    def _log_request(self, symbol: str, success: bool, net_ms: float, duration_ms: float, count: int = 1, details: str = ""):
        """
        Records ONE row per ticker fetch event.
        net_ms    = pure HTTP round-trip (request sent → response received).
        duration_ms = total fetch duration including rate-limit wait + DB saves (used for the DB log).
        count     = number of underlying API requests that fetch consumed.
        Requests = SUM(request_count), Tickers = COUNT(rows).
        """
        status_str = "SUCCESS" if success else "FAILED"

        now = time.perf_counter()
        gap_str = ""
        if self._last_fetch_end is not None:
            gap_s = round(now - self._last_fetch_end, 2)
            gap_str = f" gap:{gap_s}s"
        self._last_fetch_end = now

        net_str = f" net:{net_ms}ms"
        log_msg = f"[{self.PROVIDER_NAME.upper()}] Fetched {symbol} - {status_str}{gap_str}{net_str}"
        if details:
            log_msg += f" [{details}]"
        logger.info(log_msg)

        RequestLog.objects.create(
            provider=self.PROVIDER_NAME,
            timestamp=time.time(),
            symbol=symbol,
            duration_ms=duration_ms,
            success=success,
            request_count=max(1, int(count)),
        )
        
    def _save_daily_bars(self, symbol: str, records: list[dict]):
        """
        Saves a list of daily dictionaries to the DB.
        Expected dict format:
        {'date': 'YYYY-MM-DD', 'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}
        """
        objects_to_create = []
        objects_to_update = []
        
        # Get existing dates for this symbol to decide update vs create
        existing_days = {
            obj.date.strftime('%Y-%m-%d'): obj 
            for obj in HistoricalPrice.objects.filter(symbol=symbol)
        }
        
        for record in records:
            date_str = record['date']
            defaults = {
                'open': record['open'],
                'high': record['high'],
                'low': record['low'],
                'close': record['close'],
                'volume': record['volume']
            }
            
            if date_str in existing_days:
                existing_obj = existing_days[date_str]
                # Update existing ONLY if changed to save DB writes
                needs_update = False
                for k, v in defaults.items():
                    if getattr(existing_obj, k) != v:
                        setattr(existing_obj, k, v)
                        needs_update = True
                if needs_update:
                    objects_to_update.append(existing_obj)
            else:
                objects_to_create.append(HistoricalPrice(symbol=symbol, date=date_str, **defaults))
                
        if objects_to_create:
            HistoricalPrice.objects.bulk_create(objects_to_create, ignore_conflicts=True)
            
        if objects_to_update:
            HistoricalPrice.objects.bulk_update(objects_to_update, ['open', 'high', 'low', 'close', 'volume'])

    def _save_weekly_bars(self, symbol: str, records: list[dict]):
        """
        Saves a list of weekly dictionaries to the DB.
        Expected dict format:
        {'date': 'YYYY-MM-DD', 'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}
        """
        objects_to_create = []
        objects_to_update = []

        existing_weeks = {
            obj.date.strftime('%Y-%m-%d'): obj
            for obj in HistoricalPriceWeekly.objects.filter(symbol=symbol)
        }

        for record in records:
            date_str = record['date']
            defaults = {
                'open': record['open'],
                'high': record['high'],
                'low': record['low'],
                'close': record['close'],
                'volume': record['volume'],
            }

            if date_str in existing_weeks:
                existing_obj = existing_weeks[date_str]
                needs_update = False
                for k, v in defaults.items():
                    if getattr(existing_obj, k) != v:
                        setattr(existing_obj, k, v)
                        needs_update = True
                if needs_update:
                    objects_to_update.append(existing_obj)
            else:
                objects_to_create.append(HistoricalPriceWeekly(symbol=symbol, date=date_str, **defaults))

        if objects_to_create:
            HistoricalPriceWeekly.objects.bulk_create(objects_to_create, ignore_conflicts=True)

        if objects_to_update:
            HistoricalPriceWeekly.objects.bulk_update(objects_to_update, ['open', 'high', 'low', 'close', 'volume'])
    
    def fetch_data(self, symbol: str, force_full: bool = False):
        """
        Implement this in subclasses to perform the actual rate-limited fetching and saving.
        Should return a boolean indicating success.
        """
        raise NotImplementedError
