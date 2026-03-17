"""
Background Violations Compute Service
======================================
Runs in a daemon thread, continuously computing violations/confirmations
for all active monitored trades. Results are cached in memory so the
API endpoints can return them instantly without re-computing.
"""

import logging
import os
import threading
import time
import concurrent.futures
from datetime import date, datetime, timedelta


import pytz
from django.db import close_old_connections

from .models import MonitoredTrade, MonitorPreferences
from .violations_engine import compute_violations, precompute_heavy_data, DEFAULT_PREFERENCES

logger = logging.getLogger(__name__)

COMPUTE_INTERVAL = 0.5  # seconds between full recompute cycles
PRIMARY_BENCHMARK_TICKER = 'SPY'   # S&P 500 ETF (used for RS calculation)


class ViolationsComputeService:
    """Background service that pre-computes violations for all active trades."""

    def __init__(self):
        self.running = False
        self._thread = None
        # Cached results: {trade_id: result_dict}
        self._cache = {}
        # List of all cached results (for compute-all endpoint)
        self._all_results = []
        self._lock = threading.Lock()
        self._compute_once_lock = threading.Lock()
        self._last_compute_time = None
        # Historical data cache: {cache_key: {'data': [...], 'date': date}}
        # Refreshed once per calendar day — historical bars don't change intraday.
        self._historical_cache = {}
        # Precomputed heavy data cache: {cache_key: {'precomputed': {...}, 'date': date}}
        # MAs, trend-up, RS, since-trend maximums — refreshed once per day.
        self._precomputed_cache = {}
        # Tracks whether we already forced a one-time post-close refresh for a date.
        self._post_close_refresh_done_for = None
        # Number of currently running compute passes (background + forced recompute).
        self._compute_inflight = 0
        # Number of currently running heavy historical/precompute passes.
        self._hard_compute_inflight = 0
        # Force a hard refresh at most once per hour.
        self._last_hourly_hard_refresh_at = 0.0

    # ------------------------------------------------------------------
    # Cache access (thread-safe)
    # ------------------------------------------------------------------

    def get_result(self, trade_id):
        """Get cached result for a single trade. Returns dict or None."""
        with self._lock:
            return self._cache.get(trade_id)

    def get_all_results(self):
        """Get cached results for all active trades. Returns list of dicts."""
        with self._lock:
            return list(self._all_results)

    def get_status(self):
        """Get service status info."""
        with self._lock:
            now_ts = time.time()
            last_hourly_refresh_iso = None
            if self._last_hourly_hard_refresh_at:
                last_hourly_refresh_iso = datetime.fromtimestamp(
                    self._last_hourly_hard_refresh_at
                ).isoformat()

            return {
                'running': self.running,
                'cached_trades': len(self._cache),
                'last_compute': self._last_compute_time,
                'is_computing': self._compute_inflight > 0,
                'is_hard_computing': self._hard_compute_inflight > 0,
                'historical_cache_size': len(self._historical_cache),
                'precomputed_cache_size': len(self._precomputed_cache),
                'last_hourly_hard_refresh': last_hourly_refresh_iso,
                'post_close_refresh_done_for': (
                    str(self._post_close_refresh_done_for)
                    if self._post_close_refresh_done_for
                    else None
                ),
            }

    def get_cached_historical(self, ticker, end_date=None):
        """Return cached historical daily bars for ticker, optionally capped by end_date.

        Returns None when no cache entry exists so caller can fall back to DB.
        """
        symbol = str(ticker).upper().strip()
        with self._lock:
            entry = self._historical_cache.get(symbol)
            if not entry or not entry.get('data'):
                return None
            data = list(entry['data'])

        if end_date is not None:
            data = [row for row in data if row['date'] <= end_date]
        return data

    def _mark_compute_start(self):
        with self._lock:
            self._compute_inflight += 1

    def _mark_compute_end(self):
        with self._lock:
            self._compute_inflight = max(0, self._compute_inflight - 1)
            if self._compute_inflight == 0 and self._hard_compute_inflight > 0:
                # Fail-safe: if a hard compute flag was left active due to an exception,
                # clear it when all compute passes are finished.
                self._hard_compute_inflight = 0

    def _mark_hard_compute_start(self):
        with self._lock:
            self._hard_compute_inflight += 1

    def _mark_hard_compute_end(self):
        with self._lock:
            self._hard_compute_inflight = max(0, self._hard_compute_inflight - 1)

    def invalidate_caches(self):
        """Clear historical and precomputed caches (e.g., after trade changes)."""
        self._historical_cache.clear()
        self._precomputed_cache.clear()

    def invalidate_all_runtime_state(self):
        """Hard reset all in-memory caches/state used by compute pipelines."""
        with self._lock:
            self._cache.clear()
            self._all_results = []
            self._historical_cache.clear()
            self._precomputed_cache.clear()
            self._post_close_refresh_done_for = None

    def invalidate_ticker_caches(self, tickers):
        """Clear caches only for specific tickers, keeping others warm."""
        if not tickers:
            return

        normalized = {
            str(t).upper().strip()
            for t in tickers
            if str(t).strip()
        }
        for tkr in normalized:
            self._historical_cache.pop(tkr, None)
            self._precomputed_cache.pop(tkr, None)

    def force_recompute(self, reason='manual'):
        """Run a synchronous recompute and return fresh results immediately.
        
        Uses cached historical/precomputed data so it's fast — only the
        cheap per-day checks and realtime fetch run.
        """
        total_started = time.perf_counter()
        lock_wait_started = time.perf_counter()
        try:
            self._mark_compute_start()
            with self._compute_once_lock:
                lock_wait_ms = round((time.perf_counter() - lock_wait_started) * 1000.0, 2)
                compute_started = time.perf_counter()
                self._compute_once(reason=reason)
                compute_ms = round((time.perf_counter() - compute_started) * 1000.0, 2)
                total_ms = round((time.perf_counter() - total_started) * 1000.0, 2)
                logger.info(
                    "[ViolationsCompute][Force] mode=all reason=%s lock_wait_ms=%s compute_ms=%s total_ms=%s",
                    reason,
                    lock_wait_ms,
                    compute_ms,
                    total_ms,
                )
        except Exception as e:
            logger.error(f"Force recompute error: {e}", exc_info=True)
        finally:
            self._mark_compute_end()
        return self.get_all_results()

    def force_recompute_trade(self, trade_id, reason='manual-trade'):
        """Run a synchronous recompute for a single trade and return just that result."""
        total_started = time.perf_counter()
        lock_wait_started = time.perf_counter()
        trade_id = int(trade_id)
        try:
            self._mark_compute_start()
            with self._compute_once_lock:
                lock_wait_ms = round((time.perf_counter() - lock_wait_started) * 1000.0, 2)
                compute_started = time.perf_counter()
                self._compute_once(target_trade_ids={trade_id}, reason=reason, lightweight=True)
                compute_ms = round((time.perf_counter() - compute_started) * 1000.0, 2)
                total_ms = round((time.perf_counter() - total_started) * 1000.0, 2)
                logger.info(
                    "[ViolationsCompute][Force] mode=single trade_id=%s reason=%s lock_wait_ms=%s compute_ms=%s total_ms=%s",
                    trade_id,
                    reason,
                    lock_wait_ms,
                    compute_ms,
                    total_ms,
                )
        except Exception as e:
            logger.error("Force single-trade recompute error for trade_id=%s: %s", trade_id, e, exc_info=True)
        finally:
            self._mark_compute_end()
        return self.get_result(trade_id)

    # ------------------------------------------------------------------
    # Data fetching helpers (same as views.py but used by background thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_historical_from_server(ticker, start_date=None, end_date=None):
        """Fetch full historical daily OHLCV data directly from local DB."""
        try:
            from ticker_data.models import HistoricalPrice

            symbol = str(ticker).upper().strip()
            qs = HistoricalPrice.objects.filter(symbol=symbol)
            if start_date:
                qs = qs.filter(date__gte=start_date)
            if end_date:
                qs = qs.filter(date__lte=end_date)
            qs = qs.order_by('date')

            data = []
            for row in qs:
                data.append({
                    'date': row.date,
                    'open': float(row.open),
                    'high': float(row.high),
                    'low': float(row.low),
                    'close': float(row.close),
                    'volume': int(row.volume),
                })
            return data
        except Exception as e:
            logger.warning(f"Could not fetch historical DB data for {ticker}: {e}")
            return []

    def _get_all_realtime_data(self, tickers):
        """Get today's latest bar for the given tickers directly from the DB."""
        try:
            from ticker_data.models import HistoricalPrice
            result = {}
            for ticker in tickers:
                symbol = str(ticker).upper().strip()
                bar = (
                    HistoricalPrice.objects
                    .filter(symbol=symbol)
                    .order_by('-date')
                    .values('open', 'high', 'low', 'close', 'volume')
                    .first()
                )
                if bar:
                    vol = int(bar['volume'] or 0)
                    result[symbol] = {
                        'open': float(bar['open']),
                        'high': float(bar['high']),
                        'low': float(bar['low']),
                        'close': float(bar['close']),
                        'volume': vol,
                        'raw_volume': vol,
                    }
            return result
        except Exception as e:
            logger.warning(f"Could not get realtime data from DB: {e}")
            return {}



    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def _compute_once(self, target_trade_ids=None, reason='background', lightweight=False):
        """Run one compute cycle for all active trades or a targeted subset."""
        cycle_started = time.perf_counter()
        close_old_connections()

        target_ids = None
        if target_trade_ids:
            target_ids = {int(tid) for tid in target_trade_ids}

        now_ts = time.time()
        if (now_ts - self._last_hourly_hard_refresh_at) >= 3600:
            self.invalidate_caches()
            self._last_hourly_hard_refresh_at = now_ts
            logger.info("Hourly hard refresh triggered: historical/precomputed caches invalidated")

        trades_qs = MonitoredTrade.objects.using('violations_monitor_db').filter(is_active=True)
        if target_ids:
            trades_qs = trades_qs.filter(id__in=target_ids)
        trades = list(trades_qs)
        if not trades:
            with self._lock:
                if target_ids:
                    for trade_id in target_ids:
                        self._cache.pop(trade_id, None)
                    self._all_results = list(self._cache.values())
                else:
                    self._cache.clear()
                    self._all_results.clear()
                self._last_compute_time = datetime.now().isoformat()
            return

        # Load preferences once
        prefs_obj, _ = MonitorPreferences.objects.using(
            'violations_monitor_db'
        ).get_or_create(id=1, defaults={'preferences': DEFAULT_PREFERENCES})
        enabled = {**DEFAULT_PREFERENCES, **prefs_obj.preferences}

        today = date.today()
        with self._lock:
            previous_cache = dict(self._cache)
        new_cache = {}
        new_all = []
        did_heavy_work = False

        # --- Pre-fetch historical data per unique ticker (cached by ticker, once/day) ---
        # IMPORTANT: don't cache empty results — Flask server may not have data yet after restart
        unique_tickers = {t.ticker.upper() for t in trades}
        unique_tickers.add(PRIMARY_BENCHMARK_TICKER)   # prefer true S&P 500 index

        # Determine upfront whether this cycle needs heavy historical/precompute work.
        needs_historical_refresh = any(
            (not self._historical_cache.get(tkr)
             or self._historical_cache.get(tkr, {}).get('date') != today
             or not self._historical_cache.get(tkr, {}).get('data'))
            for tkr in unique_tickers
        )
        needs_precompute_refresh = any(
            tkr != PRIMARY_BENCHMARK_TICKER
            and (
                not self._precomputed_cache.get(tkr)
                or self._precomputed_cache.get(tkr, {}).get('date') != today
                or not self._precomputed_cache.get(tkr, {}).get('precomputed')
                or (
                    not lightweight
                    and self._precomputed_cache.get(tkr, {}).get('precomputed', {}).get('lightweight', False)
                )
            )
            for tkr in unique_tickers
        )
        hard_compute_started = False
        if needs_historical_refresh or needs_precompute_refresh:
            self._mark_hard_compute_start()
            hard_compute_started = True

        # Ensure each ticker exists in TrackedTicker so the orchestrator fetches it.
        # Write directly to DB — no HTTP call.
        t_reg = time.perf_counter()
        try:
            from ticker_data.models import TrackedTicker
            from django.utils import timezone as dj_timezone
            for tkr in unique_tickers:
                if tkr == PRIMARY_BENCHMARK_TICKER:
                    continue  # SPY is always tracked; skip to avoid unnecessary writes
                obj, created = TrackedTicker.objects.get_or_create(symbol=tkr)
                if created:
                    obj.last_trade_seen_at = dj_timezone.now()
                    obj.save(update_fields=['last_trade_seen_at'])
        except Exception as reg_err:
            logger.debug("Could not ensure ticker in TrackedTicker: %s", reg_err)
        reg_ms = round((time.perf_counter() - t_reg) * 1000, 2)

        # --- Phase 1: Historical fetch (parallel per-ticker) -----------------
        t_hist = time.perf_counter()

        def fetch_historical(tkr):
            entry = self._historical_cache.get(tkr)
            if not entry or entry['date'] != today or not entry['data']:
                data = self._get_historical_from_server(tkr)
                if data is not None and len(data) > 0:
                    return tkr, data
            return tkr, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(50, len(unique_tickers) or 1)) as executor:
            futures = {executor.submit(fetch_historical, tkr): tkr for tkr in unique_tickers}
            for future in concurrent.futures.as_completed(futures):
                tkr, data = future.result()
                if data is not None:
                    did_heavy_work = True
                    self._historical_cache[tkr] = {'data': data, 'date': today}

        hist_ms = round((time.perf_counter() - t_hist) * 1000, 2)
        hist_ready = sum(
            1
            for tkr in unique_tickers
            if self._historical_cache.get(tkr, {}).get('data')
        )
        # --- Phase 2: Precompute + realtime fetch IN PARALLEL ----------------
        # These are independent: precompute uses historical data,
        # realtime uses the Flask server's live data. Overlapping
        # them saves ~2 s per cycle when the realtime TTL expires.
        t_pre = time.perf_counter()

        sp500_data = self._historical_cache.get(PRIMARY_BENCHMARK_TICKER, {}).get('data', [])

        def precompute_for_ticker(tkr):
            pre_entry = self._precomputed_cache.get(tkr)
            tkr_data = self._historical_cache.get(tkr, {}).get('data', [])
            needs_refresh = (
                not pre_entry
                or pre_entry['date'] != today
                or not pre_entry['precomputed']
                or (
                    not lightweight
                    and pre_entry.get('precomputed', {}).get('lightweight', False)
                )
            )
            if needs_refresh:
                if tkr_data:
                    benchmark_data = sp500_data if tkr != PRIMARY_BENCHMARK_TICKER else None
                    precomputed = precompute_heavy_data(
                        tkr_data,
                        benchmark_data,
                        enabled,
                        include_risk_analysis=not lightweight,
                    )
                    return tkr, precomputed
            return tkr, None

        # Run precompute; fetch realtime from DB concurrently.
        realtime_future = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(unique_tickers) or 1) + 1) as executor:
            realtime_future = executor.submit(self._get_all_realtime_data, list(unique_tickers))
            pre_futures = {executor.submit(precompute_for_ticker, tkr): tkr for tkr in unique_tickers}
            for future in concurrent.futures.as_completed(pre_futures):
                tkr, precomputed = future.result()
                if precomputed is not None:
                    did_heavy_work = True
                    self._precomputed_cache[tkr] = {
                        'precomputed': precomputed, 'date': today,
                    }

        all_realtime = realtime_future.result()
        pre_ms = round((time.perf_counter() - t_pre) * 1000, 2)

        pre_ready = sum(
            1
            for tkr in unique_tickers
            if tkr == PRIMARY_BENCHMARK_TICKER or self._precomputed_cache.get(tkr, {}).get('precomputed')
        )
        if did_heavy_work:
            logger.debug("Heavy compute pass completed (historical/precompute refresh)")

        if hard_compute_started:
            self._mark_hard_compute_end()

        # Use realtime as "today's bar" from market open through after-hours,
        # so end-of-day values settle immediately after close.
        # Before open, we should NOT count today as up/down/engulfing/etc.
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        allow_today_realtime = now_et >= market_open and now_et.weekday() < 5

        # One-time post-close refresh: clear daily caches so finalized bars are
        # re-fetched from server as soon as market closes.
        if now_et >= market_close and now_et.weekday() < 5 and self._post_close_refresh_done_for != today:
            self.invalidate_caches()
            self._post_close_refresh_done_for = today
            logger.info(f"Post-close cache refresh triggered for {today}")

        def compute_for_trade(trade):
            try:
                ticker = trade.ticker.upper()
                start_date = trade.start_date
                end_date = (
                    trade.end_date
                    if not trade.use_latest_end_date
                    else today
                )

                daily_data = self._historical_cache.get(ticker, {}).get('data', [])
                precomputed = self._precomputed_cache.get(ticker, {}).get('precomputed')

                if not daily_data:
                    logger.debug(
                        "Skipping compute for %s (%s): no historical data yet; keeping previous cached result",
                        ticker,
                        trade.id,
                    )
                    return trade.id, None

                # --- Realtime data from bulk fetch ---
                today_rt = None
                if trade.use_latest_end_date and end_date == today and allow_today_realtime:
                    today_rt = all_realtime.get(ticker)
                benchmark_today_rt = all_realtime.get(PRIMARY_BENCHMARK_TICKER)

                result = compute_violations(
                    daily_data=daily_data,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    sp500_data=sp500_data,
                    enabled_checks=enabled,
                    today_realtime=today_rt,
                    precomputed=precomputed,
                    benchmark_today_realtime=benchmark_today_rt,
                )

                result['trade_id'] = trade.id
                result['ticker'] = ticker
                result['start_date'] = str(start_date)
                result['end_date'] = str(end_date)
                result['use_latest_end_date'] = trade.use_latest_end_date
                result['total_violations'] = len(result['violations'])
                result['total_confirmations'] = sum(
                    1 for item in result['confirmations']
                    if item.get('type') != 'inside_day'
                )

                return trade.id, result
            except Exception as e:
                logger.error(f"Error computing violations for trade {trade.id} ({trade.ticker}): {e}")
                return trade.id, None

        t_trades = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, len(trades) or 1)) as executor:
            futures = {executor.submit(compute_for_trade, trade): trade for trade in trades}
            for future in concurrent.futures.as_completed(futures):
                trade_id, result = future.result()
                if result is not None:
                    new_cache[trade_id] = result
                    new_all.append(result)
                else:
                    previous = previous_cache.get(trade_id)
                    if previous is not None:
                        preserved = dict(previous)
                        preserved['stale_preserved'] = True
                        new_cache[trade_id] = preserved
                        new_all.append(preserved)
                        logger.debug(
                            "Preserved previous result for trade_id=%s ticker=%s",
                            trade_id,
                            preserved.get('ticker'),
                        )
        trades_ms = round((time.perf_counter() - t_trades) * 1000, 2)

        with self._lock:
            if target_ids:
                merged_cache = dict(self._cache)
                for trade_id in target_ids:
                    merged_cache.pop(trade_id, None)
                merged_cache.update(new_cache)
                self._cache = merged_cache
                self._all_results = list(merged_cache.values())
            else:
                self._cache = new_cache
                self._all_results = new_all
            self._last_compute_time = datetime.now().isoformat()

        cycle_ms = round((time.perf_counter() - cycle_started) * 1000.0, 2)
        if cycle_ms >= 2000 or did_heavy_work:
            logger.info(
                "Compute cycle: mode=%s reason=%s lightweight=%s elapsed_ms=%s results=%s reg=%s hist=%s pre+rt=%s trades=%s heavy=%s",
                'single' if target_ids else 'all',
                reason,
                lightweight,
                cycle_ms,
                len(new_cache) if target_ids else len(new_all),
                reg_ms,
                hist_ms,
                pre_ms,
                trades_ms,
                did_heavy_work,
            )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self):
        logger.info("Violations compute service loop started")
        while self.running:
            try:
                self._mark_compute_start()
                with self._compute_once_lock:
                    self._compute_once()
            except Exception as e:
                logger.error(f"Violations compute service error: {e}", exc_info=True)
            finally:
                self._mark_compute_end()
            time.sleep(COMPUTE_INTERVAL)

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ViolationsComputeService"
        )
        self._thread.start()
        logger.info("Violations compute service started")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Violations compute service stopped")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_service_instance = None


def get_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = ViolationsComputeService()
    return _service_instance


def start_compute_service():
    """Start the background compute service (idempotent).

    apps.py already ensures this is only called once in the correct
    Django child process (RUN_MAIN=true), so no socket-lock guard needed.
    """
    svc = get_service()
    if svc.running:
        logger.info('Violations compute service already running — skipping duplicate start')
        return
    svc.start()
