import logging
import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote
from pathlib import Path

from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import MonitoredTrade, MonitorPreferences
from .serializers import (
    MonitoredTradeSerializer,
    MonitorPreferencesSerializer,
    MonAlertTradeSerializer,
)
from .violations_engine import ALL_CHECKS, DEFAULT_PREFERENCES, compute_violations, precompute_heavy_data

logger = logging.getLogger(__name__)

PRIMARY_BENCHMARK_TICKER = 'SPY'   # S&P 500 ETF (used for RS calculation)
CHART_SETTINGS_DB_PATH = Path(settings.BASE_DIR) / 'dbs' / 'violations_chart_settings.sqlite3'
CHART_SETTINGS_SQL_PATH = Path(settings.BASE_DIR) / 'violations_monitor' / 'sql' / 'init_violations_chart_settings.sql'
ALLOWED_SMA_SOURCES = {'close', 'open', 'high', 'low'}


def _default_sma_settings():
    return [
        {
            'length': 20,
            'r': 255,
            'g': 215,
            'b': 0,
            'opacity': 0.9,
            'thickness': 2,
            'enabled': True,
            'source': 'close',
        }
    ]


def _default_weekly_sma_settings():
    return _default_sma_settings()


def _ensure_chart_settings_db():
    CHART_SETTINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CHART_SETTINGS_DB_PATH) as conn:
        if CHART_SETTINGS_SQL_PATH.exists():
            conn.executescript(CHART_SETTINGS_SQL_PATH.read_text(encoding='utf-8'))
        else:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chart_settings (
                    ticker TEXT PRIMARY KEY,
                    sma_settings_json TEXT NOT NULL,
                    weekly_sma_settings_json TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'close',
                    highlight_marker_gap INTEGER NOT NULL DEFAULT 0,
                    open_on_bars INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chart_settings_ticker ON chart_settings(ticker);
                """
            )
        # Backward compatibility for existing DB files created before source column.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chart_settings)").fetchall()}
        if 'weekly_sma_settings_json' not in columns:
            conn.execute(
                "ALTER TABLE chart_settings ADD COLUMN weekly_sma_settings_json TEXT NOT NULL DEFAULT '[]'"
            )
        if 'source' not in columns:
            conn.execute("ALTER TABLE chart_settings ADD COLUMN source TEXT NOT NULL DEFAULT 'close'")
        if 'highlight_marker_gap' not in columns:
            conn.execute("ALTER TABLE chart_settings ADD COLUMN highlight_marker_gap INTEGER NOT NULL DEFAULT 0")
        if 'open_on_bars' not in columns:
            conn.execute("ALTER TABLE chart_settings ADD COLUMN open_on_bars INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def _normalize_source(value):
    source = str(value or 'close').lower().strip()
    return source if source in ALLOWED_SMA_SOURCES else 'close'


def _normalize_highlight_marker_gap(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(0, min(15, parsed))


def _normalize_open_on_bars(value):
    return bool(value)


def _normalize_sma_settings(items):
    normalized = []
    if not isinstance(items, list):
        return _default_sma_settings()

    for item in items[:12]:
        if not isinstance(item, dict):
            continue
        try:
            length = int(item.get('length', 20))
            r = int(item.get('r', 255))
            g = int(item.get('g', 215))
            b = int(item.get('b', 0))
            opacity = float(item.get('opacity', 0.9))
            thickness = float(item.get('thickness', 2))
            enabled = bool(item.get('enabled', True))
            source = _normalize_source(item.get('source', 'close'))
        except (TypeError, ValueError):
            continue

        normalized.append(
            {
                'length': max(2, min(400, length)),
                'r': max(0, min(255, r)),
                'g': max(0, min(255, g)),
                'b': max(0, min(255, b)),
                'opacity': max(0.05, min(1.0, opacity)),
                'thickness': max(0.1, min(8.0, round(thickness, 1))),
                'enabled': enabled,
                'source': source,
            }
        )

    return normalized or _default_sma_settings()


def _get_chart_settings_for_ticker(ticker):
    _ensure_chart_settings_db()
    tkr = str(ticker).upper().strip()
    with sqlite3.connect(CHART_SETTINGS_DB_PATH) as conn:
        row = conn.execute(
            "SELECT sma_settings_json, weekly_sma_settings_json, source, highlight_marker_gap, open_on_bars FROM chart_settings WHERE ticker = ?",
            (tkr,),
        ).fetchone()
    if not row:
        return {
            'sma_settings': _default_sma_settings(),
            'daily_sma_settings': _default_sma_settings(),
            'weekly_sma_settings': _default_weekly_sma_settings(),
            'source': 'close',
            'highlight_marker_gap': 0,
            'open_on_bars': False,
        }
    try:
        parsed_daily = json.loads(row[0])
    except Exception:
        parsed_daily = _default_sma_settings()
    try:
        parsed_weekly = json.loads(row[1])
    except Exception:
        parsed_weekly = _default_weekly_sma_settings()
    source = _normalize_source(row[2])
    normalized_daily = _normalize_sma_settings(parsed_daily)
    normalized_weekly = _normalize_sma_settings(parsed_weekly)
    return {
        'sma_settings': normalized_daily,
        'daily_sma_settings': normalized_daily,
        'weekly_sma_settings': normalized_weekly,
        'source': source,
        'highlight_marker_gap': _normalize_highlight_marker_gap(row[3]),
        'open_on_bars': _normalize_open_on_bars(row[4]),
    }


def _save_chart_settings_for_ticker(ticker, daily_sma_settings, weekly_sma_settings, source, highlight_marker_gap, open_on_bars):
    _ensure_chart_settings_db()
    tkr = str(ticker).upper().strip()
    daily_payload = json.dumps(_normalize_sma_settings(daily_sma_settings))
    weekly_payload = json.dumps(_normalize_sma_settings(weekly_sma_settings))
    normalized_source = _normalize_source(source)
    normalized_gap = _normalize_highlight_marker_gap(highlight_marker_gap)
    normalized_open_on_bars = 1 if _normalize_open_on_bars(open_on_bars) else 0
    updated_at = datetime.utcnow().isoformat()
    with sqlite3.connect(CHART_SETTINGS_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chart_settings(ticker, sma_settings_json, weekly_sma_settings_json, source, highlight_marker_gap, open_on_bars, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker)
            DO UPDATE SET
                sma_settings_json = excluded.sma_settings_json,
                weekly_sma_settings_json = excluded.weekly_sma_settings_json,
                source = excluded.source,
                highlight_marker_gap = excluded.highlight_marker_gap,
                open_on_bars = excluded.open_on_bars,
                updated_at = excluded.updated_at
            """,
            (tkr, daily_payload, weekly_payload, normalized_source, normalized_gap, normalized_open_on_bars, updated_at),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# ViewSet for MonAlertTrade CRUD
# ---------------------------------------------------------------------------
class MonAlertTradeViewSet(viewsets.ModelViewSet):
    queryset = MonitoredTrade.objects.all()
    serializer_class = MonAlertTradeSerializer

    def create(self, request, *args, **kwargs):
        """Create a trade, fetch its data directly, compute violations inline, return result."""
        flow_start = time.perf_counter()
        response = super().create(request, *args, **kwargs)
        if response.status_code == 201:
            ticker = response.data['ticker'].upper()
            trade_id = response.data.get('id')

            # SPY is already maintained locally for benchmark calculations, so avoid a redundant fetch.
            if ticker == PRIMARY_BENCHMARK_TICKER:
                fetch_trace = {'success': True, 'symbol': ticker, 'skipped': True, 'reason': 'benchmark_already_local'}
                fetch_ms = 0.0
            else:
                # --- Direct fetch (no orchestrator, fresh client, zero rate-limit contention) ---
                t0 = time.perf_counter()
                fetch_trace = _instant_fetch(ticker)
                fetch_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # --- Inline compute ---
            t1 = time.perf_counter()
            trade_result, _ = _compute_trade_immediate(trade_id)
            compute_ms = round((time.perf_counter() - t1) * 1000.0, 2)

            # Sync compute service cache so background polls see the new result immediately
            from .compute_service import get_service
            svc = get_service()
            svc.invalidate_ticker_caches([ticker])
            if trade_result is None:
                trade_result = svc.force_recompute_trade(trade_id, reason='add_trade_fallback')

            total_ms = round((time.perf_counter() - flow_start) * 1000.0, 2)
            response.data['initial_result'] = trade_result
            logger.info(
                "[Add] ticker=%s fetch_ms=%.0f compute_ms=%.0f total_ms=%.0f success=%s skipped_fetch=%s",
                ticker, fetch_ms, compute_ms, total_ms,
                fetch_trace.get('success'),
                fetch_trace.get('skipped', False),
            )
        return response

    def destroy(self, request, *args, **kwargs):
        """Delete a monitored trade and purge ticker data when no active trades remain for that ticker."""
        instance = self.get_object()
        ticker = str(instance.ticker or '').upper().strip()

        response = super().destroy(request, *args, **kwargs)

        if ticker:
            cleanup_trace = _cleanup_ticker_storage_if_unmonitored(ticker)
            logger.info(
                "[MonitoredTradeViewSet][Delete] ticker=%s cleanup=%s",
                ticker,
                cleanup_trace,
            )

        return response


# Backward-compatible symbol alias
MonitoredTradeViewSet = MonAlertTradeViewSet


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------
@api_view(['GET', 'PUT'])
def preferences_view(request):
    """Get or update monitor preferences (singleton)."""
    obj, _ = MonitorPreferences.objects.using('violations_monitor_db').get_or_create(
        id=1, defaults={'preferences': DEFAULT_PREFERENCES}
    )

    if request.method == 'GET':
        # Merge in any new checks that don't exist in saved prefs
        prefs = {**DEFAULT_PREFERENCES, **obj.preferences}
        return Response({'preferences': prefs, 'all_checks': {
            k: {'name': v[0], 'type': v[1], 'severity': v[2]}
            for k, v in ALL_CHECKS.items()
        }})

    if request.method == 'PUT':
        prefs = request.data.get('preferences', {})
        obj.preferences = prefs
        obj.save(using='violations_monitor_db')
        return Response({'preferences': obj.preferences})


# ---------------------------------------------------------------------------
# Compute violations endpoint (reads from background cache)
# ---------------------------------------------------------------------------
@api_view(['GET'])
def compute_violations_view(request, trade_id):
    """Return pre-computed violations/confirmations for a specific trade.
    
    Results are computed continuously by the background ViolationsComputeService
    and cached in memory. This endpoint just reads the cache — instant response.
    """
    from .compute_service import get_service

    req_started = time.perf_counter()
    try:
        MonitoredTrade.objects.using('violations_monitor_db').get(id=trade_id)
    except MonitoredTrade.DoesNotExist:
        return Response({'error': 'Trade not found'}, status=status.HTTP_404_NOT_FOUND)

    result = get_service().get_result(trade_id)
    if result is None:
        logger.info(
            "[ViolationsCompute][Single] trade_id=%s cache_hit=false elapsed_ms=%.2f",
            trade_id,
            (time.perf_counter() - req_started) * 1000.0,
        )
        return Response(
            {'error': 'Violations not yet computed. The background service is still processing.'},
            status=status.HTTP_202_ACCEPTED,
        )
    logger.info(
        "[ViolationsCompute][Single] trade_id=%s cache_hit=true elapsed_ms=%.2f violations=%s confirmations=%s",
        trade_id,
        (time.perf_counter() - req_started) * 1000.0,
        result.get('total_violations', 0),
        result.get('total_confirmations', 0),
    )
    return Response(result)


# ---------------------------------------------------------------------------
# Compute violations for ALL active trades (reads from background cache)
# ---------------------------------------------------------------------------
@api_view(['GET', 'POST'])
def compute_all_violations_view(request):
    """Return violations for all active monitored trades.
    
    GET  — read from background cache (instant, used for polling).
    POST — force a synchronous recompute then return fresh results
           (used after date changes so results are immediate).
    """
    from .compute_service import get_service

    req_started = time.perf_counter()
    svc = get_service()
    if request.method == 'POST':
        results = svc.force_recompute()
        logger.info(
            "[ViolationsCompute][All] mode=force count=%s elapsed_ms=%.2f",
            len(results),
            (time.perf_counter() - req_started) * 1000.0,
        )
    else:
        results = svc.get_all_results()
        logger.info(
            "[ViolationsCompute][All] mode=cache count=%s elapsed_ms=%.2f",
            len(results),
            (time.perf_counter() - req_started) * 1000.0,
        )
    return Response(results)


@api_view(['GET', 'PUT'])
def chart_settings_view(request, ticker):
    """Get or update persisted chart SMA settings for a ticker.

    Stored in a dedicated SQLite DB file managed by a SQL script, not Django migrations.
    """
    tkr = str(ticker).upper().strip()

    if request.method == 'GET':
        settings_payload = _get_chart_settings_for_ticker(tkr)
        return Response(
            {
                'ticker': tkr,
                'sma_settings': settings_payload['sma_settings'],
                'daily_sma_settings': settings_payload['daily_sma_settings'],
                'weekly_sma_settings': settings_payload['weekly_sma_settings'],
                'highlight_marker_gap': settings_payload['highlight_marker_gap'],
                'open_on_bars': settings_payload['open_on_bars'],
            }
        )

    settings_payload = _get_chart_settings_for_ticker(tkr)
    incoming_daily = request.data.get('daily_sma_settings', request.data.get('sma_settings', []))
    incoming_weekly = request.data.get('weekly_sma_settings', settings_payload['weekly_sma_settings'])
    incoming_gap = request.data.get('highlight_marker_gap', 0)
    incoming_open_on_bars = request.data.get('open_on_bars', None)
    source = settings_payload['source']
    normalized_daily = _normalize_sma_settings(incoming_daily)
    normalized_weekly = _normalize_sma_settings(
        settings_payload['weekly_sma_settings'] if incoming_weekly is None else incoming_weekly
    )
    normalized_gap = _normalize_highlight_marker_gap(incoming_gap)
    normalized_open_on_bars = settings_payload['open_on_bars'] if incoming_open_on_bars is None else _normalize_open_on_bars(incoming_open_on_bars)
    _save_chart_settings_for_ticker(tkr, normalized_daily, normalized_weekly, source, normalized_gap, normalized_open_on_bars)
    return Response({
        'ticker': tkr,
        'sma_settings': normalized_daily,
        'daily_sma_settings': normalized_daily,
        'weekly_sma_settings': normalized_weekly,
        'highlight_marker_gap': normalized_gap,
        'open_on_bars': normalized_open_on_bars,
    })


@api_view(['POST'])
def compute_session_violations_view(request, trade_id):
    """Compute violations for a monitored trade using temporary start/end dates.

    This endpoint does not persist any changes. It is intended for fast chart-hover
    interactions where start/end are adjusted in the UI session only.
    """
    req_started = time.perf_counter()

    try:
        trade = MonitoredTrade.objects.using('violations_monitor_db').get(id=trade_id)
    except MonitoredTrade.DoesNotExist:
        return Response({'error': 'Trade not found'}, status=status.HTTP_404_NOT_FOUND)

    def _parse_date(value):
        if value in (None, ''):
            return None
        if isinstance(value, date):
            return value
        try:
            return datetime.strptime(str(value), '%Y-%m-%d').date()
        except Exception:
            return None

    use_latest_payload = request.data.get('use_latest_end_date', None)
    if use_latest_payload is None:
        use_latest_end_date = bool(trade.use_latest_end_date)
    else:
        use_latest_end_date = bool(use_latest_payload)

    start_override = _parse_date(request.data.get('start_date'))
    end_override = _parse_date(request.data.get('end_date'))

    start_date = start_override or trade.start_date
    today = date.today()
    if use_latest_end_date:
        end_date = today
    else:
        end_date = end_override or trade.end_date or today

    if start_date > end_date:
        return Response(
            {'error': 'start_date cannot be after end_date'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    ticker = trade.ticker.upper().strip()
    # Prefer warm in-memory cache from compute service to avoid SQLite lock waits
    # under heavy write load from background fetch/orchestrator.
    from .compute_service import get_service
    svc = get_service()

    # Cap historical input at selected end date so old-window sessions do not
    # process years of future bars.
    daily_data = svc.get_cached_historical(ticker, end_date=end_date)
    if daily_data is None:
        daily_data = _get_historical_from_server(ticker, end_date=end_date)
    if not daily_data:
        return Response(
            {'error': f'No historical data available for {ticker}'},
            status=status.HTTP_404_NOT_FOUND,
        )

    if ticker == PRIMARY_BENCHMARK_TICKER:
        sp500_data = daily_data
    else:
        sp500_data = svc.get_cached_historical(PRIMARY_BENCHMARK_TICKER, end_date=end_date)
        if sp500_data is None:
            sp500_data = _get_historical_from_server(PRIMARY_BENCHMARK_TICKER, end_date=end_date)
    prefs_obj, _ = MonitorPreferences.objects.using('violations_monitor_db').get_or_create(
        id=1,
        defaults={'preferences': DEFAULT_PREFERENCES},
    )
    enabled = {**DEFAULT_PREFERENCES, **prefs_obj.preferences}

    t_pre = time.perf_counter()
    precomputed = precompute_heavy_data(
        daily_data,
        sp500_data,
        enabled,
        include_risk_analysis=False,
    )
    pre_ms = round((time.perf_counter() - t_pre) * 1000.0, 2)

    t_compute = time.perf_counter()
    result = compute_violations(
        daily_data=daily_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        sp500_data=sp500_data,
        enabled_checks=enabled,
        today_realtime=None,
        precomputed=precomputed,
    )
    compute_ms = round((time.perf_counter() - t_compute) * 1000.0, 2)

    result['trade_id'] = trade.id
    result['ticker'] = ticker
    result['start_date'] = str(start_date)
    result['end_date'] = str(end_date)
    result['use_latest_end_date'] = use_latest_end_date
    result['total_violations'] = len(result.get('violations', []))
    result['total_confirmations'] = sum(
        1 for item in result.get('confirmations', []) if item.get('type') != 'inside_day'
    )
    result['session_only'] = True

    elapsed_ms = round((time.perf_counter() - req_started) * 1000.0, 2)
    logger.info(
        "[ViolationsCompute][Session] trade_id=%s ticker=%s start=%s end=%s latest=%s pre_ms=%s compute_ms=%s elapsed_ms=%s",
        trade.id,
        ticker,
        start_date,
        end_date,
        use_latest_end_date,
        pre_ms,
        compute_ms,
        elapsed_ms,
    )

    return Response(result)


@api_view(['GET'])
def compute_status_view(request):
    """Return background compute service status."""
    from .compute_service import get_service

    svc = get_service()
    status_data = svc.get_status()
    status_data['message'] = 'Computing...' if status_data.get('is_hard_computing') else 'Idle'
    return Response(status_data)


@api_view(['GET'])
def compute_health_view(request):
    """Return diagnostic health snapshot for compute refresh/caches."""
    from .compute_service import get_service

    svc = get_service()
    status_data = svc.get_status()
    status_data['message'] = 'Computing...' if status_data.get('is_hard_computing') else 'Idle'
    return Response(status_data)


# ---------------------------------------------------------------------------
# Manual data refresh
# ---------------------------------------------------------------------------
@api_view(['POST'])
def refresh_historical_data(request):
    """Force refresh historical data for a ticker."""
    ticker = request.data.get('ticker', '').upper()
    if not ticker:
        return Response({'error': 'ticker required'}, status=status.HTTP_400_BAD_REQUEST)

    _invalidate_historical_in_server(ticker)
    return Response({
        'message': f'Invalidated cached data for {ticker}. '
                   f'Fresh data will be fetched by the server in the next round-robin pass.',
        'ticker': ticker,
    })


# ---------------------------------------------------------------------------
# Historical data info
# ---------------------------------------------------------------------------
@api_view(['GET'])
def historical_data_info(request, ticker):
    """Get info about cached historical data for a ticker from the Flask server."""
    ticker = ticker.upper()
    records = _get_historical_from_server(ticker)
    count = len(records)
    if count == 0:
        return Response({'ticker': ticker, 'count': 0, 'first_date': None, 'last_date': None})
    return Response({
        'ticker': ticker,
        'count': count,
        'first_date': str(records[0]['date']),
        'last_date': str(records[-1]['date']),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_historical_from_server(ticker, start_date=None, end_date=None):
    """Fetch historical daily OHLCV data from the new local ticker_data Django DB.
    
    Returns list of dicts with date (datetime.date), open, high, low, close, volume.
    """
    from ticker_data.models import HistoricalPrice
    try:
        queryset = HistoricalPrice.objects.filter(symbol=ticker.upper().strip())
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)
            
        # We need it sorted chronologically (oldest first) for our computations
        queryset = queryset.order_by('date')
        
        data = []
        for obj in queryset:
            data.append({
                'date': obj.date,
                'open': obj.open,
                'high': obj.high,
                'low': obj.low,
                'close': obj.close,
                'volume': obj.volume
            })
        return data
    except Exception as e:
        logger.warning(f"Could not fetch historical data for {ticker} from local DB: {e}")
        return []


def _add_ticker_to_server(ticker):
    from ticker_data.models import TrackedTicker
    TrackedTicker.objects.get_or_create(symbol=ticker.upper().strip())

def _ensure_ticker_server_running():
    # Deprecated. Handled by the run_price_orchestrator management command.
    pass

def _invalidate_historical_in_server(ticker):
    # Deprecated. Historical cache is now direct DB query.
    pass

def _warmup_historical_in_server(ticker):
    # Deprecated. We now hit the InstantFetchView over HTTP.
    pass


def _instant_fetch(ticker: str) -> dict:
    """
    Fetch full history for one ticker directly — no orchestrator, no shared rate-limiter.
    Each call gets a brand-new client instance so it fires immediately.
    """
    from ticker_data.models import ProviderSettings, TrackedTicker
    from ticker_data.services.yfinance_client import YFinanceClient
    from ticker_data.services.webull_client import get_shared_webull_client
    from django.utils import timezone

    started = time.perf_counter()
    symbol = ticker.upper().strip()
    try:
        ticker_obj, created = TrackedTicker.objects.get_or_create(symbol=symbol)
        ticker_obj.last_trade_seen_at = timezone.now()
        ticker_obj.save(update_fields=['last_trade_seen_at'])

        provider = ProviderSettings.load().active_provider

        if provider == 'yfinance':
            # Fresh instance = independent rate-limiter window, fires immediately
            client = YFinanceClient()
        else:
            client = get_shared_webull_client()

        success = client.fetch_data(symbol, force_full=True)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.info("[InstantFetch] symbol=%s provider=%s success=%s elapsed_ms=%.0f", symbol, provider, success, elapsed_ms)
        return {'success': success, 'symbol': symbol, 'provider': provider, 'elapsed_ms': elapsed_ms}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.error("[InstantFetch] symbol=%s failed elapsed_ms=%.0f error=%s", symbol, elapsed_ms, exc, exc_info=True)
        return {'success': False, 'symbol': symbol, 'elapsed_ms': elapsed_ms, 'error': str(exc)}


def _cleanup_ticker_storage_if_unmonitored(ticker: str) -> dict:
    """Remove ticker from ticker_data tracking and delete its stored bars if no active monitored trades remain."""
    normalized = str(ticker or '').upper().strip()
    if not normalized:
        return {'removed': False, 'reason': 'empty_ticker'}

    still_monitored = MonitoredTrade.objects.using('violations_monitor_db').filter(
        ticker__iexact=normalized,
        is_active=True,
    ).exists()
    if still_monitored:
        return {'removed': False, 'reason': 'still_monitored'}

    try:
        from ticker_data.models import TrackedTicker, HistoricalPrice, HistoricalPrice5m, HistoricalPriceWeekly

        tracked_deleted, _ = TrackedTicker.objects.filter(symbol=normalized).delete()
        daily_deleted, _ = HistoricalPrice.objects.filter(symbol=normalized).delete()
        m5_deleted, _ = HistoricalPrice5m.objects.filter(symbol=normalized).delete()
        weekly_deleted, _ = HistoricalPriceWeekly.objects.filter(symbol=normalized).delete()

        try:
            from .compute_service import get_service

            svc = get_service()
            svc.invalidate_ticker_caches([normalized])
            # Force a lightweight refresh so deleted ticker drops quickly from cached all-results.
            svc.force_recompute(reason='trade_deleted_cleanup')
        except Exception as cache_err:
            logger.warning("Could not refresh compute caches after cleanup for %s: %s", normalized, cache_err)

        return {
            'removed': True,
            'ticker': normalized,
            'tracked_deleted': tracked_deleted,
            'daily_deleted': daily_deleted,
            'm5_deleted': m5_deleted,
            'weekly_deleted': weekly_deleted,
        }
    except Exception as exc:
        logger.error("Failed ticker storage cleanup for %s: %s", normalized, exc, exc_info=True)
        return {'removed': False, 'reason': 'cleanup_error', 'error': str(exc)}


def _compute_trade_immediate(trade_id):
    """Compute one trade result inline from local ticker_data DB for instant add-response."""
    started = time.perf_counter()
    try:
        from ticker_data.models import HistoricalPrice
        from .violations_engine import compute_violations, precompute_heavy_data

        trade = MonitoredTrade.objects.using('violations_monitor_db').get(id=trade_id)
        ticker = trade.ticker.upper()
        today = date.today()
        end_date = trade.end_date if not trade.use_latest_end_date else today

        prefs_obj, _ = MonitorPreferences.objects.using('violations_monitor_db').get_or_create(
            id=1,
            defaults={'preferences': DEFAULT_PREFERENCES},
        )
        enabled = {**DEFAULT_PREFERENCES, **prefs_obj.preferences}

        t_daily = time.perf_counter()
        daily_rows = list(HistoricalPrice.objects.filter(symbol=ticker).order_by('date'))
        daily_data = [
            {
                'date': row.date,
                'open': row.open,
                'high': row.high,
                'low': row.low,
                'close': row.close,
                'volume': row.volume,
            }
            for row in daily_rows
        ]
        daily_ms = round((time.perf_counter() - t_daily) * 1000.0, 2)

        if not daily_data:
            trace = {
                'success': False,
                'trade_id': trade_id,
                'ticker': ticker,
                'reason': 'no_daily_data',
                'daily_ms': daily_ms,
                'elapsed_ms': round((time.perf_counter() - started) * 1000.0, 2),
            }
            logger.warning(
                "[MonitoredTradeViewSet][DirectCompute] no daily data trade_id=%s ticker=%s daily_ms=%s",
                trade_id,
                ticker,
                daily_ms,
            )
            return None, trace

        t_spy = time.perf_counter()
        if ticker == PRIMARY_BENCHMARK_TICKER:
            sp500_data = daily_data
        else:
            spy_rows = list(HistoricalPrice.objects.filter(symbol=PRIMARY_BENCHMARK_TICKER).order_by('date'))
            sp500_data = [
                {
                    'date': row.date,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume,
                }
                for row in spy_rows
            ]
        spy_ms = round((time.perf_counter() - t_spy) * 1000.0, 2)

        t_pre = time.perf_counter()
        precomputed = precompute_heavy_data(
            daily_data,
            sp500_data,
            enabled,
            include_risk_analysis=True,
        )
        pre_ms = round((time.perf_counter() - t_pre) * 1000.0, 2)

        t_compute = time.perf_counter()
        result = compute_violations(
            daily_data=daily_data,
            ticker=ticker,
            start_date=trade.start_date,
            end_date=end_date,
            sp500_data=sp500_data,
            enabled_checks=enabled,
            today_realtime=None,
            precomputed=precomputed,
        )
        compute_ms = round((time.perf_counter() - t_compute) * 1000.0, 2)

        result['trade_id'] = trade.id
        result['ticker'] = ticker
        result['start_date'] = str(trade.start_date)
        result['end_date'] = str(end_date)
        result['use_latest_end_date'] = trade.use_latest_end_date
        result['total_violations'] = len(result['violations'])
        result['total_confirmations'] = sum(
            1 for item in result['confirmations'] if item.get('type') != 'inside_day'
        )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        trace = {
            'success': True,
            'trade_id': trade_id,
            'ticker': ticker,
            'daily_rows': len(daily_data),
            'spy_rows': len(sp500_data),
            'daily_ms': daily_ms,
            'spy_ms': spy_ms,
            'precompute_ms': pre_ms,
            'compute_ms': compute_ms,
            'elapsed_ms': elapsed_ms,
        }
        logger.info(
            "[MonitoredTradeViewSet][DirectCompute] done trade_id=%s ticker=%s rows=%s spy_rows=%s daily_ms=%s spy_ms=%s pre_ms=%s compute_ms=%s elapsed_ms=%s",
            trade_id,
            ticker,
            len(daily_data),
            len(sp500_data),
            daily_ms,
            spy_ms,
            pre_ms,
            compute_ms,
            elapsed_ms,
        )
        return result, trace
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.error(
            "[MonitoredTradeViewSet][DirectCompute] failed trade_id=%s elapsed_ms=%s error=%s",
            trade_id,
            elapsed_ms,
            e,
            exc_info=True,
        )
        return None, {
            'success': False,
            'trade_id': trade_id,
            'error': str(e),
            'elapsed_ms': elapsed_ms,
        }
