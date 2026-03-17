import time
import requests
import logging
from datetime import datetime, timedelta
from django.utils import timezone
from ..models import ProviderSettings, TrackedTicker, HistoricalPrice, HistoricalPrice5m, HistoricalPriceWeekly
from .yfinance_client import YFinanceClient
from .webull_client import WebullClient, get_shared_webull_client

logger = logging.getLogger(__name__)

def _purge_price_data(old_provider: str, new_provider: str):
    """Delete all price data and invalidate caches.  Used on every provider switch."""
    try:
        db_alias = 'ticker_data_db'
        daily_del, _ = HistoricalPrice.objects.using(db_alias).all().delete()
        weekly_del, _ = HistoricalPriceWeekly.objects.using(db_alias).all().delete()
        logger.info(
            "Auto-purged price data on provider switch %s -> %s: daily=%s weekly=%s",
            old_provider, new_provider, daily_del, weekly_del,
        )
        try:
            from violations_monitor.compute_service import get_service
            get_service().invalidate_all_runtime_state()
        except Exception:
            pass
    except Exception as purge_err:
        logger.error("Auto-purge on provider switch failed: %s", purge_err)


class Orchestrator:
    # How many seconds of continuous failure before auto-swapping providers
    FAILOVER_WINDOW_SECONDS = 600.0  # 10 minutes — ignore short network hiccups
    # Guardrails to detect when only short-window data exists locally.
    MIN_READY_DAILY_BARS = 30
    MIN_READY_WEEKLY_BARS_SPY = 50
    SPY_ENSURE_INTERVAL_SECONDS = 30.0
    ORPHAN_CLEANUP_INTERVAL_SECONDS = 60.0

    def __init__(self):
        self.yfinance = YFinanceClient()
        self.webull = get_shared_webull_client()
        self.trade_server_status_url = "http://localhost:5002/status"

        # Unified failure tracking for the active provider
        self._consecutive_failures = 0
        self._first_failure_ts: float | None = None
        self._last_webull_not_connected_log_ts = 0.0
        self._last_spy_ensure_ts = 0.0
        self._last_orphan_cleanup_ts = 0.0

    def run_loop(self):
        """
        The main infinite loop that runs as a Django management command.
        """
        # Delay startup to allow Django to finish initializing and avoid AppConfig DB access warnings
        time.sleep(3)
        logger.info("Starting Price Orchestrator loop...")
        
        while True:
            try:
                self._run_single_iteration()
            except Exception as e:
                logger.error(f"Orchestrator loop crashed: {e}")
                time.sleep(5)  # Pause before retrying
                
    def _run_single_iteration(self):
        self._ensure_spy_tracked()
        # Timer-driven cleanup: keep only actively monitored symbols (+system tickers).
        self._cleanup_unmonitored_ticker_data()
        settings = ProviderSettings.load()
        tickers = list(TrackedTicker.objects.all().order_by('added_at'))
        interval_seconds = self._get_interval_seconds(settings)
        
        if not tickers:
            time.sleep(1)
            return
            
        # Strict stale-data cleanup by row insertion age.
        self._cleanup_old_data()
        
        # Periodically prune inactive tickers
        self._prune_inactive_tickers(tickers)

        current_provider = settings.active_provider
        active_client = self.yfinance if current_provider == 'yfinance' else self.webull

        # --- pre-flight: webull must be connected to be usable ----------
        if current_provider == 'webull':
            webull_status = self.webull.get_status()
            if webull_status.get('status') != 'connected':
                now = time.time()
                if now - self._last_webull_not_connected_log_ts >= 5.0:
                    self._last_webull_not_connected_log_ts = now
                    logger.warning(
                        "[Orchestrator] Webull selected but not connected (status=%s, login_in_progress=%s). "
                        "Skipping fetch cycle and waiting for login.",
                        webull_status.get('status'),
                        webull_status.get('login_in_progress'),
                    )
                # Count this as a failure window for auto-swap
                self._record_failure()
                if self._should_failover():
                    self._do_failover(settings)
                else:
                    time.sleep(interval_seconds)
                return

        # --- round-robin fetch ------------------------------------------
        for ticker_obj in tickers:
            symbol = ticker_obj.symbol.upper()
            force_full_needed = self._needs_full_refresh(symbol)

            if force_full_needed:
                logger.info(
                    "[Orchestrator] No local price data for %s. Running force_full fetch.",
                    symbol,
                )

            success = active_client.fetch_data(symbol, force_full=force_full_needed)

            if success and force_full_needed:
                logger.info(
                    "[Orchestrator] Initial full refresh completed for %s",
                    symbol,
                )

            # --- unified bidirectional auto-fallback --------------------
            if success:
                self._reset_failure_tracking()
            else:
                self._record_failure()
                if self._should_failover():
                    self._do_failover(settings)
                    return  # restart iteration with new provider

    # ------------------------------------------------------------------ #
    #  Bidirectional auto-failover helpers                                #
    # ------------------------------------------------------------------ #

    def _record_failure(self):
        """Note a fetch failure.  Starts the failure window clock on the first one."""
        self._consecutive_failures += 1
        if self._first_failure_ts is None:
            self._first_failure_ts = time.time()

    def _reset_failure_tracking(self):
        self._consecutive_failures = 0
        self._first_failure_ts = None

    def _should_failover(self) -> bool:
        """True when failures have persisted for >= FAILOVER_WINDOW_SECONDS."""
        if self._first_failure_ts is None or self._consecutive_failures < 2:
            return False
        return (time.time() - self._first_failure_ts) >= self.FAILOVER_WINDOW_SECONDS

    def _do_failover(self, settings: ProviderSettings):
        """Switch to the other provider, purge price data, persist, and reset counters."""
        old = settings.active_provider
        new = 'webull' if old == 'yfinance' else 'yfinance'

        # If falling over to webull, only do it if webull is actually connected
        if new == 'webull':
            ws = self.webull.get_status()
            if ws.get('status') != 'connected':
                logger.info(
                    "[Orchestrator] Would failover to Webull but it is not connected (%s). Staying on %s.",
                    ws.get('status'), old,
                )
                # Reset so we don't spam this check every iteration
                self._reset_failure_tracking()
                return

        logger.warning(
            "[Orchestrator] Auto-failover %s -> %s after %.0fs of failures (%d consecutive).",
            old, new,
            time.time() - (self._first_failure_ts or time.time()),
            self._consecutive_failures,
        )

        _purge_price_data(old, new)

        settings.active_provider = new
        settings.save()
        self._reset_failure_tracking()

    def _get_interval_seconds(self, settings: ProviderSettings) -> float:
        try:
            max_requests_per_10s = float(settings.max_requests_per_10s)
            if max_requests_per_10s <= 0:
                return 0.5
            return max(10.0 / max_requests_per_10s, 0.2)
        except Exception:
            return 0.5

    def _ensure_spy_tracked(self):
        """Periodically ensure SPY is present in tracked tickers even if removed manually."""
        now = time.time()
        if (now - self._last_spy_ensure_ts) < self.SPY_ENSURE_INTERVAL_SECONDS:
            return
        self._last_spy_ensure_ts = now
        try:
            TrackedTicker.objects.get_or_create(symbol='SPY')
        except Exception as exc:
            logger.debug("[Orchestrator] Could not ensure SPY in tracked tickers: %s", exc)

    def _cleanup_old_data(self):
        """Delete any price rows that were ingested more than 1 day ago."""
        now = timezone.now()
        one_day_ago_dt = now - timedelta(days=1)

        deleted_daily_old, _ = HistoricalPrice.objects.filter(ingested_at__lt=one_day_ago_dt).delete()
        deleted_weekly_old, _ = HistoricalPriceWeekly.objects.filter(ingested_at__lt=one_day_ago_dt).delete()

        if deleted_daily_old > 0 or deleted_weekly_old > 0:
            logger.warning(
                "[Orchestrator] Strict stale purge by ingested_at (>1 day) deleted_daily_rows=%s deleted_weekly_rows=%s cutoff_ts=%s",
                deleted_daily_old,
                deleted_weekly_old,
                one_day_ago_dt.isoformat(),
            )

        remaining_daily_old = HistoricalPrice.objects.filter(ingested_at__lt=one_day_ago_dt).count()
        remaining_weekly_old = HistoricalPriceWeekly.objects.filter(ingested_at__lt=one_day_ago_dt).count()
        if remaining_daily_old > 0 or remaining_weekly_old > 0:
            logger.error(
                "[Orchestrator] Stale-data verification failed: remaining_daily_rows=%s remaining_weekly_rows=%s",
                remaining_daily_old,
                remaining_weekly_old,
            )

    def _needs_full_refresh(self, symbol):
        """Return True when local data is missing or too shallow for steady-state fast fetches."""
        symbol = symbol.upper()
        daily_count = HistoricalPrice.objects.filter(symbol=symbol).count()
        weekly_count = HistoricalPriceWeekly.objects.filter(symbol=symbol).count() if symbol == 'SPY' else self.MIN_READY_WEEKLY_BARS_SPY

        needs_refresh = (
            daily_count < self.MIN_READY_DAILY_BARS
            or weekly_count < self.MIN_READY_WEEKLY_BARS_SPY
        )

        if needs_refresh:
            logger.info(
                "[Orchestrator] Force-full required for %s (daily_count=%s, weekly_count=%s, thresholds=%s/%s)",
                symbol,
                daily_count,
                weekly_count,
                self.MIN_READY_DAILY_BARS,
                self.MIN_READY_WEEKLY_BARS_SPY,
            )

        return needs_refresh


    def _prune_inactive_tickers(self, db_tickers):
        """Removes tickers from tracking if not in an active trade and > 8 hours old."""
        # Simple placeholder for the logic from server.py 
        # that checks the buy_seller application.
        pass

    def _cleanup_unmonitored_ticker_data(self):
        """Every minute, delete ticker_data rows for symbols not in active violations monitoring."""
        now = time.time()
        if (now - self._last_orphan_cleanup_ts) < self.ORPHAN_CLEANUP_INTERVAL_SECONDS:
            return
        self._last_orphan_cleanup_ts = now

        try:
            from violations_monitor.models import MonitoredTrade

            active_symbols = {
                str(t).upper().strip()
                for t in MonitoredTrade.objects.using('violations_monitor_db')
                .filter(is_active=True)
                .values_list('ticker', flat=True)
                if str(t).strip()
            }

            # Keep SPY because violations/RS computations use it as benchmark input.
            keep_symbols = set(active_symbols)
            keep_symbols.add('SPY')

            tracked_symbols = {
                str(t).upper().strip()
                for t in TrackedTicker.objects.values_list('symbol', flat=True)
                if str(t).strip()
            }
            daily_symbols = {
                str(s).upper().strip()
                for s in HistoricalPrice.objects.values_list('symbol', flat=True).distinct()
                if str(s).strip()
            }
            m5_symbols = {
                str(s).upper().strip()
                for s in HistoricalPrice5m.objects.values_list('symbol', flat=True).distinct()
                if str(s).strip()
            }
            weekly_symbols = {
                str(s).upper().strip()
                for s in HistoricalPriceWeekly.objects.values_list('symbol', flat=True).distinct()
                if str(s).strip()
            }

            all_symbols = tracked_symbols | daily_symbols | m5_symbols | weekly_symbols
            orphan_symbols = sorted(all_symbols - keep_symbols)
            if not orphan_symbols:
                return

            tracked_deleted, _ = TrackedTicker.objects.exclude(symbol__in=list(keep_symbols)).delete()
            daily_deleted, _ = HistoricalPrice.objects.exclude(symbol__in=list(keep_symbols)).delete()
            m5_deleted, _ = HistoricalPrice5m.objects.exclude(symbol__in=list(keep_symbols)).delete()
            weekly_deleted, _ = HistoricalPriceWeekly.objects.exclude(symbol__in=list(keep_symbols)).delete()

            try:
                from violations_monitor.compute_service import get_service

                svc = get_service()
                svc.invalidate_ticker_caches(orphan_symbols)
                svc.force_recompute(reason='timer_orphan_cleanup')
            except Exception as cache_err:
                logger.warning("[Orchestrator] Orphan cleanup cache refresh failed: %s", cache_err)

            logger.info(
                "[Orchestrator] Orphan cleanup removed symbols=%s tracked_deleted=%s daily_deleted=%s m5_deleted=%s weekly_deleted=%s keep_symbols=%s",
                len(orphan_symbols),
                tracked_deleted,
                daily_deleted,
                m5_deleted,
                weekly_deleted,
                len(keep_symbols),
            )
        except Exception as exc:
            logger.error("[Orchestrator] Orphan cleanup failed: %s", exc, exc_info=True)


def force_fetch_ticker_now(symbol: str, reason: str = "manual", force_full: bool | None = None, period: str | None = None):
    """Force an immediate synchronous fetch for one ticker and return timing trace metadata."""
    started = time.perf_counter()
    normalized_symbol = str(symbol).upper().strip()
    if not normalized_symbol:
        return {
            'success': False,
            'symbol': normalized_symbol,
            'error': 'Invalid symbol',
            'elapsed_ms': 0.0,
        }

    try:
        ticker_obj, created = TrackedTicker.objects.get_or_create(symbol=normalized_symbol)
        ticker_obj.last_trade_seen_at = timezone.now()
        ticker_obj.save(update_fields=['last_trade_seen_at'])

        settings = ProviderSettings.load()
        provider = settings.active_provider
        client = YFinanceClient() if provider == 'yfinance' else get_shared_webull_client()
        effective_force_full = created if force_full is None else bool(force_full)
        if effective_force_full:
            fetch_profile_daily = 'max' if normalized_symbol == 'SPY' else '10y'
        else:
            fetch_profile_daily = '5d'

        logger.info(
            "[Orchestrator][InstantFetch] start symbol=%s reason=%s provider=%s force_full=%s created=%s profile=daily:%s",
            normalized_symbol,
            reason,
            provider,
            effective_force_full,
            created,
            fetch_profile_daily,
        )

        fetch_started = time.perf_counter()
        if isinstance(client, YFinanceClient):
            success = bool(client.fetch_data(normalized_symbol, force_full=effective_force_full, period_override=period))
        else:
            # Webull uses bar counts; map common period strings to approximate counts
            _PERIOD_TO_BARS = {'1y': 260, '2y': 520, '3y': 780, '5y': 1200}
            wb_count = min(_PERIOD_TO_BARS.get(period, 1200), 1200) if period else 1200
            success = bool(client.fetch_data(normalized_symbol, force_full=effective_force_full, count_override=wb_count))
        fetch_ms = round((time.perf_counter() - fetch_started) * 1000.0, 2)

        metrics_started = time.perf_counter()
        daily_qs = HistoricalPrice.objects.filter(symbol=normalized_symbol)
        daily_count = daily_qs.count()
        latest_daily = daily_qs.order_by('-date').values_list('date', flat=True).first()
        metrics_ms = round((time.perf_counter() - metrics_started) * 1000.0, 2)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.info(
            "[Orchestrator][InstantFetch] done symbol=%s success=%s provider=%s force_full=%s elapsed_ms=%s fetch_ms=%s metrics_ms=%s daily_count=%s latest_daily=%s",
            normalized_symbol,
            success,
            provider,
            effective_force_full,
            elapsed_ms,
            fetch_ms,
            metrics_ms,
            daily_count,
            latest_daily,
        )

        return {
            'success': success,
            'symbol': normalized_symbol,
            'provider': provider,
            'created': created,
            'force_full': effective_force_full,
            'fetch_ms': fetch_ms,
            'metrics_ms': metrics_ms,
            'elapsed_ms': elapsed_ms,
            'daily_count': daily_count,
            'daily_latest': str(latest_daily) if latest_daily else None,
            'reason': reason,
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.error(
            "[Orchestrator][InstantFetch] failed symbol=%s reason=%s elapsed_ms=%s error=%s",
            normalized_symbol,
            reason,
            elapsed_ms,
            exc,
            exc_info=True,
        )
        return {
            'success': False,
            'symbol': normalized_symbol,
            'reason': reason,
            'error': str(exc),
            'elapsed_ms': elapsed_ms,
        }
