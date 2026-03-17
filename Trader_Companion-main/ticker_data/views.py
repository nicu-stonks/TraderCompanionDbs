from django.http import JsonResponse
from rest_framework.views import APIView
from .models import TrackedTicker, ProviderSettings, HistoricalPrice, HistoricalPrice5m, HistoricalPriceWeekly
from .services.yfinance_client import YFinanceClient
from .services.webull_client import WebullClient, get_shared_webull_client
from .services.orchestrator import force_fetch_ticker_now, _purge_price_data
from django.utils import timezone
from datetime import timezone as dt_timezone
from django.db import transaction
from django.db.utils import OperationalError
import logging
import time
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import RequestLog

logger = logging.getLogger(__name__)


def _run_with_sqlite_lock_retry(fn, retries=3, sleep_seconds=0.15):
    """Retry transient SQLite lock errors a few times before failing."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except OperationalError as exc:
            if 'database is locked' not in str(exc).lower():
                raise
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(sleep_seconds)
            else:
                raise
    if last_exc:
        raise last_exc

class InstantFetchView(APIView):
    """
    Synchronously forces a high-priority fetch of a ticker's data, bypassing the background orchestrator loop.
    Returns immediately after the DB is written to.
    """
    def post(self, request):
        symbol = request.data.get('symbol')
        if not symbol:
            return JsonResponse({'error': 'Symbol is required'}, status=400)

        symbol = symbol.upper().strip()
        logger.info("[InstantFetchView] Received instant fetch request for %s", symbol)

        trace = force_fetch_ticker_now(symbol=symbol, reason='ticker_data.fetch_now')
        if trace.get('success'):
            return JsonResponse({
                'status': 'success',
                'symbol': symbol,
                'message': 'Data fetched and saved to DB.',
                'newly_tracked': trace.get('created', False),
                'trace': trace,
            })
        return JsonResponse({
            'status': 'error',
            'symbol': symbol,
            'message': 'Failed to fetch data.',
            'trace': trace,
        }, status=500)


@api_view(['GET', 'PUT'])
def provider_settings_view(request):
    settings = _run_with_sqlite_lock_retry(lambda: ProviderSettings.load())
    if request.method == 'PUT':
        new_limit = request.data.get('max_requests_per_10s')
        if new_limit is not None:
            settings.max_requests_per_10s = float(new_limit)
        
        new_provider = request.data.get('active_provider')
        if new_provider in ['yfinance', 'webull']:
            old_provider = settings.active_provider
            settings.active_provider = new_provider

            # Auto-purge price data when switching providers so stale data
            # from the old provider doesn't mix with the new one.
            if old_provider != new_provider:
                _purge_price_data(old_provider, new_provider)

        _run_with_sqlite_lock_retry(lambda: settings.save())
        
    interval_ms = int((10.0 / settings.max_requests_per_10s) * 1000) if settings.max_requests_per_10s > 0 else 500
    return Response({
        'max_requests_per_10s': settings.max_requests_per_10s,
        'active_provider': settings.active_provider,
        'interval_ms': interval_ms
    })

@api_view(['GET', 'POST'])
def request_interval_view(request):
    """
    Compatibility endpoint for old dataFetcherAPI.ts requests to /request-interval.
    Mapped seamlessly into max_requests_per_10s.
    """
    settings = ProviderSettings.load()
    if request.method == 'POST':
        interval_ms = request.data.get('interval_ms')
        if interval_ms:
            # interval of 500ms means 2 per sec, means 20 per 10s.
            # Convert ms -> per 10s: (1000 / ms) * 10
            settings.max_requests_per_10s = (1000.0 / float(interval_ms)) * 10.0
            settings.save()
            
    current_interval_ms = int((10.0 / settings.max_requests_per_10s) * 1000) if settings.max_requests_per_10s > 0 else 500
    return Response({
        'interval_ms': current_interval_ms
    })

@api_view(['GET'])
def request_stats_view(request):
    now = time.time()
    # One row per ticker fetch event; request_count holds the API calls that fetch consumed.
    logs = list(RequestLog.objects.filter(timestamp__gte=now - 10.0, success=True)
                .values('timestamp', 'request_count'))

    logs_1s = [l for l in logs if l['timestamp'] >= now - 1.0]
    logs_5s = [l for l in logs if l['timestamp'] >= now - 5.0]

    # Tickers = row count, Requests = sum of request_count
    tickers_last_1s  = len(logs_1s)
    tickers_last_5s  = len(logs_5s)
    tickers_last_10s = len(logs)

    req_last_1s  = sum(l['request_count'] for l in logs_1s)
    req_last_5s  = sum(l['request_count'] for l in logs_5s)
    req_last_10s = sum(l['request_count'] for l in logs)

    tickers_avg_10s = round(tickers_last_10s / 10.0, 2)
    req_avg_10s     = round(req_last_10s     / 10.0, 2)

    est_seconds_per_ticker  = round(1.0 / tickers_avg_10s, 2) if tickers_avg_10s > 0 else None
    est_requests_per_ticker = round(req_last_10s / tickers_last_10s, 2) if tickers_last_10s > 0 else None

    # Loop time = how long until a given ticker gets refreshed again.
    # Equals (number of tracked tickers) × (time per ticker in the round-robin loop).
    tracked_ticker_count = TrackedTicker.objects.count()
    est_loop_seconds = round(est_seconds_per_ticker * tracked_ticker_count, 2) if est_seconds_per_ticker and tracked_ticker_count > 0 else None

    return Response({
        'last_1s': req_last_1s,
        'last_5s': req_last_5s,
        'last_10s': req_last_10s,
        'per_second_avg_5s': round(req_last_5s / 5.0, 2),
        'per_second_avg_10s': req_avg_10s,
        'tickers_last_1s': tickers_last_1s,
        'tickers_last_5s': tickers_last_5s,
        'tickers_last_10s': tickers_last_10s,
        'tickers_per_second_avg_5s': round(tickers_last_5s / 5.0, 2),
        'tickers_per_second_avg_10s': tickers_avg_10s,
        'est_seconds_per_ticker': est_seconds_per_ticker,
        'est_requests_per_ticker': est_requests_per_ticker,
        'est_loop_seconds': est_loop_seconds,
        'tracked_ticker_count': tracked_ticker_count,
    })

@api_view(['GET'])
def ticker_errors_view(request):
    now = time.time()
    # Recently failed requests (last 10 minutes)
    failed_logs = RequestLog.objects.filter(timestamp__gte=now - 600.0, success=False).order_by('symbol', '-timestamp')
    
    errors = {}
    for log in failed_logs:
        if log.symbol and log.symbol not in errors:
            errors[log.symbol] = {
                'message': f"Failed to fetch data from {log.provider}",
                'timestamp': timezone.datetime.fromtimestamp(log.timestamp, tz=dt_timezone.utc).isoformat(),
                'type': 'FetchError'
            }
            
    return Response({
        'errors': errors,
        'count': len(errors)
    })

@api_view(['GET', 'POST', 'DELETE'])
def tickers_view(request, symbol=None):
    if request.method == 'DELETE':
        if symbol:
            _run_with_sqlite_lock_retry(lambda: TrackedTicker.objects.filter(symbol=symbol.upper()).delete())
            return Response({'status': 'success', 'message': f'Deleted {symbol}'})
        else:
            _run_with_sqlite_lock_retry(lambda: TrackedTicker.objects.all().delete())
            return Response({'status': 'success', 'message': 'Deleted all tickers'})

    if request.method == 'POST':
        sym = request.data.get('symbol')
        if sym:
            sym = sym.upper().strip()
            _run_with_sqlite_lock_retry(lambda: TrackedTicker.objects.get_or_create(symbol=sym))
            return Response({'status': 'success', 'symbol': sym})
        return Response({'error': 'Symbol missing'}, status=400)

    tickers = _run_with_sqlite_lock_retry(lambda: list(TrackedTicker.objects.values_list('symbol', flat=True)))
    return Response({
        'tickers': tickers,
        'total_count': len(tickers)
    })


from .models import HistoricalPrice
from .services.webull_client import WebullClient, get_shared_webull_client


def _get_webull_client():
    return get_shared_webull_client()

@api_view(['GET'])
def get_webull_status(request):
    client = _get_webull_client()
    return Response(client.get_status())

@api_view(['POST'])
def start_webull_login(request):
    client = _get_webull_client()
    return Response(client.start_login())

@api_view(['GET'])
def get_all_latest_data(request):
    tickers = TrackedTicker.objects.values_list('symbol', flat=True)
    result = {}
    for symbol in tickers:
        bars = list(HistoricalPrice.objects.filter(symbol=symbol).order_by('-date')[:2])
        if not bars:
            continue
        latest = bars[0]
        prev = bars[1] if len(bars) > 1 else None
        
        result[symbol] = {
            'currentPrice': latest.close,
            'dayOpen': latest.open,
            'dayHigh': latest.high,
            'dayLow': latest.low,
            'volume': latest.volume,
            'previousClose': prev.close if prev else latest.open
        }
    return Response({'tickers': result, 'count': len(result)})

@api_view(['GET'])
def get_latest_data(request, symbol):
    symbol = symbol.upper()
    bars = list(HistoricalPrice.objects.filter(symbol=symbol).order_by('-date')[:2])
    if not bars:
        return Response({'error': 'no data'}, status=404)
    latest = bars[0]
    prev = bars[1] if len(bars) > 1 else None
    return Response({
        'currentPrice': latest.close,
        'dayOpen': latest.open,
        'dayHigh': latest.high,
        'dayLow': latest.low,
        'volume': latest.volume,
        'previousClose': prev.close if prev else latest.open
    })

from .models import ProviderSettings, TrackedTicker
from django.utils import timezone
from datetime import timedelta, time as dt_time
from zoneinfo import ZoneInfo

def is_market_open_sync():
    try:
        now = timezone.now().astimezone(ZoneInfo("US/Eastern"))
        if now.weekday() >= 5:
            return False, "Market is closed (Weekend)"
        open_time = dt_time(9, 30)
        close_time = dt_time(16, 0)
        if open_time <= now.time() <= close_time:
            return True, "Market is open"
        
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.time() > close_time:
            next_open += timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
            
        diff = next_open - now
        hours, remainder = divmod(diff.seconds, 3600)
        minutes = remainder // 60
        return False, f"Market opens in {hours}h {minutes}m"
    except Exception as e:
        return True, "Market is open (fallback)"

@api_view(['GET'])
def get_server_status(request):
    is_open, msg = is_market_open_sync()
    tickers_count = TrackedTicker.objects.count()
    settings = ProviderSettings.get_settings()
    interval_ms = 1000.0 * (10.0 / settings.max_requests_per_10s) if settings.max_requests_per_10s > 0 else 0
    interval_sec = 10.0 / settings.max_requests_per_10s if settings.max_requests_per_10s > 0 else 0
    return Response({
        'running': True,
        'market_open': is_open,
        'market_message': msg,
        'current_time': str(timezone.now()),
        'tickers_count': tickers_count,
        'request_interval_seconds': interval_sec,
        'request_interval_ms': interval_ms
    })

from .models import HistoricalPrice5m

@api_view(['GET'])
def get_historical_data(request, symbol):
    symbol = symbol.upper()
    bars = HistoricalPrice.objects.filter(symbol=symbol).order_by('date')
    if not bars.exists():
        return Response({'error': 'no data'}, status=404)
        
    data = []
    for b in bars:
        data.append({
            'date': str(b.date),
            'open': float(b.open),
            'high': float(b.high),
            'low': float(b.low),
            'close': float(b.close),
            'volume': int(b.volume)
        })
    return Response({'symbol': symbol, 'count': len(data), 'data': data})

@api_view(['GET'])
def get_historical_5m_data(request, symbol):
    symbol = symbol.upper()
    bars = HistoricalPrice5m.objects.filter(symbol=symbol).order_by('-timestamp')[:4000]
    if not bars.exists():
        return Response({'error': 'no data'}, status=404)
        
    data = []
    for b in bars:
        data.append({
            'timestamp': str(b.timestamp),
            'open': float(b.open),
            'high': float(b.high),
            'low': float(b.low),
            'close': float(b.close),
            'volume': int(b.volume)
        })
    data.reverse()
    return Response({'symbol': symbol, 'count': len(data), 'data': data})


def _aggregate_weekly_from_daily_rows(daily_rows):
    """Aggregate chronological daily OHLCV rows into ISO-week OHLCV bars."""
    weekly = []
    current = None

    for row in daily_rows:
        d = row.date
        week_key = d.isocalendar()[:2]  # (iso_year, iso_week)

        if current is None or current['_key'] != week_key:
            if current is not None:
                weekly.append(current)
            current = {
                '_key': week_key,
                'date': d,
                'open': float(row.open),
                'high': float(row.high),
                'low': float(row.low),
                'close': float(row.close),
                'volume': int(row.volume),
            }
            continue

        current['high'] = max(current['high'], float(row.high))
        current['low'] = min(current['low'], float(row.low))
        current['close'] = float(row.close)
        current['volume'] += int(row.volume)

    if current is not None:
        weekly.append(current)

    return [
        {
            'date': str(item['date']),
            'open': item['open'],
            'high': item['high'],
            'low': item['low'],
            'close': item['close'],
            'volume': item['volume'],
        }
        for item in weekly
    ]


@api_view(['GET'])
def get_historical_weekly_data(request, symbol):
    symbol = symbol.upper()
    bars = HistoricalPriceWeekly.objects.filter(symbol=symbol).order_by('date')
    if bars.exists():
        data = []
        for b in bars:
            data.append({
                'date': str(b.date),
                'open': float(b.open),
                'high': float(b.high),
                'low': float(b.low),
                'close': float(b.close),
                'volume': int(b.volume)
            })
        return Response({'symbol': symbol, 'count': len(data), 'timeframe': 'weekly', 'data': data})

    # Fallback: derive weekly candles from daily bars when explicit weekly rows
    # are not available (common for non-SPY symbols).
    daily_rows = HistoricalPrice.objects.filter(symbol=symbol).order_by('date')
    if not daily_rows.exists():
        return Response({'error': 'no data'}, status=404)

    data = _aggregate_weekly_from_daily_rows(daily_rows)
    logger.info(
        "[TickerData] Weekly fallback aggregation used for %s (daily_rows=%s, weekly_rows=0, weekly_rows_built=%s)",
        symbol,
        daily_rows.count(),
        len(data),
    )
    return Response({'symbol': symbol, 'count': len(data), 'timeframe': 'weekly', 'data': data})


@api_view(['DELETE'])
def purge_all_price_data(request):
    """
    Forcefully delete ALL historical price data (daily + 5m) for every ticker.
    Tracked tickers are preserved so re-fetching starts automatically.
    """
    try:
        db_alias = 'ticker_data_db'
        daily_qs = HistoricalPrice.objects.using(db_alias)
        fivemin_qs = HistoricalPrice5m.objects.using(db_alias)
        weekly_qs = HistoricalPriceWeekly.objects.using(db_alias)

        daily_before = daily_qs.count()
        fivemin_before = fivemin_qs.count()
        weekly_before = weekly_qs.count()

        with transaction.atomic(using=db_alias):
            daily_deleted, _ = daily_qs.delete()
            fivemin_deleted, _ = fivemin_qs.delete()
            weekly_deleted, _ = weekly_qs.delete()

        daily_after = HistoricalPrice.objects.using(db_alias).count()
        fivemin_after = HistoricalPrice5m.objects.using(db_alias).count()
        weekly_after = HistoricalPriceWeekly.objects.using(db_alias).count()

        cache_reset = False
        try:
            from violations_monitor.compute_service import get_service
            get_service().invalidate_all_runtime_state()
            cache_reset = True
        except Exception as cache_err:
            logger.warning(f"Price purge succeeded but cache reset failed: {cache_err}")

        logger.info(
            "Purged ticker_data_db price data: daily_before=%s daily_deleted=%s daily_after=%s "
            "fivemin_before=%s fivemin_deleted=%s fivemin_after=%s "
            "weekly_before=%s weekly_deleted=%s weekly_after=%s cache_reset=%s",
            daily_before,
            daily_deleted,
            daily_after,
            fivemin_before,
            fivemin_deleted,
            fivemin_after,
            weekly_before,
            weekly_deleted,
            weekly_after,
            cache_reset,
        )
        return Response({
            'success': True,
            'database': db_alias,
            'deleted_daily': daily_deleted,
            'deleted_5m': fivemin_deleted,
            'deleted_weekly': weekly_deleted,
            'daily_before': daily_before,
            'daily_after': daily_after,
            'm5_before': fivemin_before,
            'm5_after': fivemin_after,
            'weekly_before': weekly_before,
            'weekly_after': weekly_after,
            'cache_reset': cache_reset,
        })
    except Exception as e:
        logger.error(f"Error purging price data: {e}")
        return Response({'error': str(e)}, status=500)
