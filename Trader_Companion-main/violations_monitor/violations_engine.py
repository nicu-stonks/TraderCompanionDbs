"""
Violations & Confirmations Engine
=================================
Computes trading violations (red) and confirmations (green) for a stock
over a given date range, using daily OHLCV data.

Terminology:
- "trend up start": the date where the 200-day SMA began trending upward.
  Determined by checking 200MA at 20-day intervals; more recent must be >= 0.5% above older.
- "violation": a bearish / unhealthy signal (shown in red)
- "confirmation": a bullish / healthy signal (shown in green)
- "warning": an orange-severity signal (shown in orange, counted as violation)

All data lists are expected to be sorted by date ascending.
Each row is a dict: {date, open, high, low, close, volume}
"""

from datetime import date, timedelta
from collections import defaultdict
from bisect import bisect_left, bisect_right
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sma(values, window):
    """Return list of SMA values (None where not enough data)."""
    out = [None] * len(values)
    if window <= 0 or len(values) < window:
        return out
    running = sum(values[:window])
    out[window - 1] = running / window
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out[i] = running / window
    return out


def _pct_change(old, new):
    """Percentage change from old to new."""
    if old == 0 or old is None:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


def _day_range_pct(row):
    """Percentage spread of the day: (high - low) / low * 100."""
    if row['low'] == 0:
        return 0.0
    return (row['high'] - row['low']) / row['low'] * 100.0


def _percentile_rank(sorted_values, value):
    """Percentile rank (0-100) of value within sorted ascending values."""
    if not sorted_values:
        return 0.0
    return (bisect_right(sorted_values, value) / len(sorted_values)) * 100.0


def _normal_pdf(x, mean, std):
    """Normal distribution PDF value."""
    if std <= 0:
        return 0.0
    z = (x - mean) / std
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2 * math.pi))


def _silverman_bandwidth(values, std):
    """Bandwidth for KDE using Silverman's rule-of-thumb."""
    n = len(values)
    if n <= 1:
        return 1e-6

    sorted_vals = sorted(values)
    q1 = sorted_vals[int(0.25 * (n - 1))]
    q3 = sorted_vals[int(0.75 * (n - 1))]
    iqr = q3 - q1

    robust_sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if robust_sigma <= 0:
        robust_sigma = max(abs(sorted_vals[-1] - sorted_vals[0]) / 6.0, 1e-6)

    return max(0.9 * robust_sigma * (n ** (-1.0 / 5.0)), 1e-6)


def _kde_density(x, values, bandwidth):
    """Gaussian KDE density estimate at x."""
    if not values or bandwidth <= 0:
        return 0.0

    inv_nh = 1.0 / (len(values) * bandwidth)
    norm_const = 1.0 / math.sqrt(2 * math.pi)
    acc = 0.0
    for v in values:
        z = (x - v) / bandwidth
        acc += math.exp(-0.5 * z * z) * norm_const
    return acc * inv_nh


def _build_risk_distribution(values, bins=161, symmetric=True, min_x=None, max_x=None, split_at='mean', exceed_inclusive=False):
    """Build KDE-shaped distribution + split-point tail probabilities."""
    if not values:
        return {
            'sample_size': 0,
            'mean': 0.0,
            'median': 0.0,
            'q1': 0.0,
            'q3': 0.0,
            'p10': 0.0,
            'p90': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'bins': [],
        }

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    min_v = min(values)
    max_v = max(values)

    if symmetric:
        max_abs = max(abs(min_v), abs(max_v), 0.5)
        if std > 0:
            max_abs = max(max_abs, 3.0 * std)
        domain_min = -max_abs if min_x is None else min_x
        domain_max = max_abs if max_x is None else max_x
        fallback_span = max_abs if max_abs > 0 else 1.0
    else:
        domain_min = min_v if min_x is None else min_x
        domain_max = max_v if max_x is None else max_x
        if domain_max < domain_min:
            domain_max = domain_min
        fallback_span = max(domain_max - domain_min, 1.0)

    if bins < 3:
        bins = 3

    if domain_max == domain_min:
        domain_max = domain_min + fallback_span

    step = (domain_max - domain_min) / (bins - 1)
    xs = [(domain_min + i * step) for i in range(bins)]

    std_for_pdf = std if std > 1e-9 else max((domain_max - domain_min) / 6.0, 1e-6)
    bandwidth = _silverman_bandwidth(values, std_for_pdf)
    raw_pdf = [_kde_density(x, values, bandwidth) for x in xs]
    peak_pdf = max(raw_pdf) if raw_pdf else 1.0
    if peak_pdf <= 0:
        peak_pdf = 1.0

    sorted_vals = sorted(values)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    q1 = sorted_vals[int(0.25 * (n - 1))]
    q3 = sorted_vals[int(0.75 * (n - 1))]
    p10 = sorted_vals[int(0.10 * (n - 1))]
    p90 = sorted_vals[int(0.90 * (n - 1))]
    split_point = 0.0 if split_at == 'zero' else mean
    built_bins = []
    for x, pdf_val in zip(xs, raw_pdf):
        cdf = bisect_right(sorted_vals, x) / n
        if exceed_inclusive:
            exceed = (n - bisect_left(sorted_vals, x)) / n  # inclusive greater-or-equal
        else:
            exceed = (n - bisect_right(sorted_vals, x)) / n  # strict greater-than

        # Left side of split point: cumulative probability up to x.
        # Right side of split point: exceedance probability from x upward.
        if x <= split_point:
            tail = cdf
        else:
            tail = exceed

        built_bins.append({
            'x': round(x, 2),
            'density_pct': round((pdf_val / peak_pdf) * 100.0, 2),
            'cdf_pct': round(cdf * 100.0, 2),
            'exceed_pct': round(exceed * 100.0, 2),
            'tail_pct': round(tail * 100.0, 2),
        })

    return {
        'sample_size': n,
        'mean': round(mean, 3),
        'median': round(median, 3),
        'q1': round(q1, 3),
        'q3': round(q3, 3),
        'p10': round(p10, 3),
        'p90': round(p90, 3),
        'split_point': round(split_point, 3),
        'std': round(std, 3),
        'min': round(min_v, 3),
        'max': round(max_v, 3),
        'bins': built_bins,
    }


def compute_risk_analysis(daily_data, lookback_days=252):
    """Compute one-year risk distributions for gap, day % change, and spread magnitude."""
    if not daily_data or len(daily_data) < 2:
        return {
            'lookback_days': lookback_days,
            'sample_size': 0,
            'as_of_date': None,
            'gap_risk': _build_risk_distribution([]),
            'daily_change': _build_risk_distribution([]),
            'daily_spread': _build_risk_distribution([]),
        }

    end_idx = len(daily_data) - 1
    start_idx = max(1, end_idx - lookback_days + 1)

    gap_values = []
    day_change_values = []
    spread_values = []

    for i in range(start_idx, end_idx + 1):
        row = daily_data[i]
        prev_close = daily_data[i - 1]['close']

        gap_values.append(_pct_change(prev_close, row['open']))
        day_change_values.append(_pct_change(prev_close, row['close']))

        spread_values.append(_day_range_pct(row))

    sample_size = len(day_change_values)
    as_of_date = str(daily_data[end_idx]['date']) if daily_data else None

    return {
        'lookback_days': lookback_days,
        'sample_size': sample_size,
        'as_of_date': as_of_date,
        'gap_risk': _build_risk_distribution(gap_values, symmetric=True, split_at='zero'),
        'daily_change': _build_risk_distribution(day_change_values, symmetric=True, split_at='zero'),
        'daily_spread': _build_risk_distribution(spread_values, symmetric=False, min_x=0.0, split_at='zero', exceed_inclusive=True),
    }


def _is_up_day(row, prev_close):
    """True if this day's close > previous close."""
    return row['close'] > prev_close


def _is_down_day(row, prev_close):
    """True if this day's close < previous close."""
    return row['close'] < prev_close


def _close_in_upper_pct(row, pct=0.10):
    """True if close is in the upper `pct` of the day's range."""
    rng = row['high'] - row['low']
    if rng == 0:
        return True
    return row['close'] >= row['high'] - pct * rng


def _close_in_lower_pct(row, pct=0.10):
    """True if close is in the lower `pct` of the day's range."""
    rng = row['high'] - row['low']
    if rng == 0:
        return False
    return row['close'] <= row['low'] + pct * rng


def _aggregate_weeks(daily):
    """Aggregate daily data into weekly OHLCV (Mon-Fri groups)."""
    weeks = []
    if not daily:
        return weeks
    current_week = None
    for row in daily:
        d = row['date']
        # iso weekday: Mon=1 .. Sun=7; use isocalendar week number
        iso = d.isocalendar()
        week_key = (iso[0], iso[1])
        if current_week is None or current_week['_key'] != week_key:
            if current_week is not None:
                weeks.append(current_week)
            current_week = {
                '_key': week_key,
                'date': d,  # first day of the week in data
                'end_date': d,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
            }
        else:
            current_week['high'] = max(current_week['high'], row['high'])
            current_week['low'] = min(current_week['low'], row['low'])
            current_week['end_date'] = d
            current_week['close'] = row['close']
            current_week['volume'] += row['volume']
    if current_week is not None:
        weeks.append(current_week)
    return weeks


# ---------------------------------------------------------------------------
# Trend-up detection
# ---------------------------------------------------------------------------

def find_trend_up_start(dates, closes, window=200, interval=20, threshold=0.005):
    """
    Walk backwards from the most recent day in 20-day steps.
    At each step, check if 200MA(current) >= 200MA(current - 20) * (1 + 0.5%).
    The earliest point where this holds continuously is the trend-up start.
    Returns (index, date) or (None, None).
    """
    ma200 = _sma(closes, window)
    # Start from the last valid MA200 point
    end_idx = len(closes) - 1
    if ma200[end_idx] is None:
        return None, None

    trend_start_idx = end_idx
    idx = end_idx
    while idx - interval >= 0:
        newer = ma200[idx]
        older = ma200[idx - interval]
        if newer is None or older is None:
            break
        if newer >= older * (1 + threshold):
            trend_start_idx = idx - interval
            idx -= interval
        else:
            break

    return trend_start_idx, dates[trend_start_idx]


# ---------------------------------------------------------------------------
# RS Line calculation – stock_price / SP500_price
# MarketSurge "RS New High/Low" uses this simple relative performance line.
# ---------------------------------------------------------------------------

def _calc_rs_series(stock_closes, stock_dates, sp_closes, sp_dates):
    """
    Calculate the RS Line as stock_price / sp500_price for each date
    where both have data. This is the line MarketSurge uses for "RS New High"
    / "RS New Low" detection.

    Returns dict: {date: rs_line_value}
    """
    sp_map = dict(zip(sp_dates, sp_closes))
    rs_series = {}

    for i, d in enumerate(stock_dates):
        sp_close = sp_map.get(d)
        if sp_close is not None and sp_close > 0:
            rs_series[d] = stock_closes[i] / sp_close

    return rs_series


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

# Complete list of check keys with metadata
ALL_CHECKS = {
    # key: (display_name, type: violation/confirmation/info, severity: red/green/orange/gray)
    'closes_on_high':           ('Closes on High (within 10%)', 'confirmation', 'green'),
    'closes_on_low':            ('Closes on Low (within 10%)', 'violation', 'red'),
    'down_up_largest_vol':      ('Down/Up Largest Vol (since trend up)', 'violation', 'red'),
    'down_50pct_vol_increase':  ('Down on 50% Volume Increase (vs 50-day MA)', 'violation', 'red'),
    'largest_pct_down_high_vol':('Largest % Down Day on High Vol Day (since trend up)', 'violation', 'red'),
    'largest_down_day_pct':     ('Largest Down Day % (since trend up)', 'violation', 'red'),
    'largest_down_week_high_vol':('Largest Down Week on High Vol (since trend up)', 'violation', 'red'),
    'largest_pct_down_week':    ('Largest Price % Down Week (since trend up)', 'violation', 'red'),
    'bearish_engulfing':        ('Bearish Engulfing Day', 'violation', 'red'),
    'bullish_engulfing':        ('Bullish Engulfing Day', 'confirmation', 'green'),
    'squat_reversal':           ('Squat on Buy Day', 'violation', 'red'),
    'rs_new_high':              ('RS New High', 'confirmation', 'green'),
    'rs_new_low':               ('RS New Low', 'violation', 'red'),
    'up_30pct_vol_increase':    ('Up on 30%+ Vol Increase (vs 50-day MA)', 'confirmation', 'green'),
    'widest_spread':            ('Widest Spread (since trend up)', 'violation', 'red'),
    'inside_day':               ('Inside Day', 'confirmation', 'green'),
    'close_below_200ma':        ('Close Below 200MA', 'violation', 'red'),
    'close_above_200ma':        ('Close Above 200MA', 'confirmation', 'green'),
    'largest_gap':              ('Largest Gap (since trend up)', 'violation', 'red'),
    'extended_warning':         ('Extended Warning', 'violation', 'red'),
    'very_extended':            ('Very Extended', 'violation', 'red'),
    'days_up_down':             ('Days Up / Days Down', 'info', 'gray'),
    'weekly_lower_lows':        ('N Weekly Lower Lows', 'violation', 'red'),
    'weekly_higher_highs':      ('N Weekly Higher Highs', 'confirmation', 'green'),
    'daily_megaphone':          ('2+ Day Megaphone (engulfing)', 'violation', 'red'),
    'daily_lower_lows':         ('N Daily Lower Lows', 'violation', 'red'),
    'daily_higher_highs':       ('N Daily Higher Highs', 'confirmation', 'green'),
    'big_down_day':              ('Big Down Day', 'violation', 'red'),
    'big_up_day':                ('Big Up Day', 'confirmation', 'green'),
    'above_20day_20pct':        ('20%+ Above 20-day MA', 'violation', 'red'),
    'angle_d_warning':          ('Angle D Warning (50MA downtrend)', 'violation', 'red'),
    'gap_alert':                ('Gap Alert (3%+ gap)', 'violation', 'red'),
    'large_squat_reversal':     ('Large Reversal', 'violation', 'red'),
    'ants':                     ('ANTS (12 of 15 days up)', 'confirmation', 'green'),
    'good_bad_close':           ('Good Close / Bad Close', 'info', 'gray'),
    'close_below_10ma':         ('Close Below 10MA', 'violation', 'red'),
    'close_below_20ma':         ('Close Below 20MA', 'violation', 'red'),
    'close_below_50ma':         ('Close Below 50MA', 'violation', 'red'),
    'close_above_50ma':         ('Close Above 50MA', 'confirmation', 'green'),
    'close_below_50ma_high_vol':('Close Below 50MA on High Volume', 'violation', 'red'),
}

DEFAULT_PREFERENCES = {k: True for k in ALL_CHECKS}

# Extended-state thresholds.
EXTENDED_WARNING_PERCENTILE = 0.75
VERY_EXTENDED_PERCENTILE = 0.90
MIN_EXTENDED_LOOKBACK_DAYS = 252


def precompute_heavy_data(daily_data, sp500_data=None, enabled_checks=None, include_risk_analysis=True):
    """
    Pre-compute expensive data that only changes when a new daily bar is added.
    Call once per day per (ticker, date-range) and reuse across all 1-second
    compute cycles.  Returns a dict to pass as *precomputed* to
    ``compute_violations()``, or None if daily_data is empty.
    """
    if not daily_data:
        return None

    if enabled_checks is None:
        enabled_checks = DEFAULT_PREFERENCES

    closes  = [r['close']  for r in daily_data]
    opens   = [r['open']   for r in daily_data]
    volumes = [r['volume'] for r in daily_data]
    dates   = [r['date']   for r in daily_data]

    # Moving averages
    ma10     = _sma(closes, 10)
    ma20     = _sma(closes, 20)
    ma50     = _sma(closes, 50)
    ma200    = _sma(closes, 200)
    vol_ma50 = _sma(volumes, 50)

    # Trend-up detection
    trend_start_idx, trend_start_date = find_trend_up_start(dates, closes)

    # RS (Relative Strength) series
    rs_series = {}
    if sp500_data and (enabled_checks.get('rs_new_high', True)
                       or enabled_checks.get('rs_new_low', True)):
        sp_dates  = [r['date']  for r in sp500_data]
        sp_closes = [r['close'] for r in sp500_data]
        rs_series = _calc_rs_series(closes, dates, sp_closes, sp_dates)

    risk_analysis = compute_risk_analysis(daily_data, lookback_days=252) if include_risk_analysis else None

    return {
        'ma10': ma10, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
        'vol_ma50': vol_ma50,
        'trend_start_idx': trend_start_idx,
        'trend_start_date': trend_start_date,
        'rs_series': rs_series,
        'lightweight': not include_risk_analysis,
        'risk_analysis': risk_analysis,
    }


def compute_violations(daily_data, ticker, start_date, end_date,
                        sp500_data=None, enabled_checks=None,
                        today_realtime=None, precomputed=None,
                        benchmark_today_realtime=None):
    """
    Main entry point. Computes all violations and confirmations.

    Parameters
    ----------
    daily_data : list[dict]
        Full historical daily OHLCV for the ticker, sorted by date asc.
        Each dict: {date: date, open: float, high, low, close, volume: int}
        Should include data well before start_date for MA/trend calculations.
    ticker : str
    start_date : date
    end_date : date
    sp500_data : list[dict] or None
        Historical daily OHLCV for SPY, same format.
    enabled_checks : dict or None
        Map of check_key -> bool. None means all enabled.
    today_realtime : dict or None
        If provided, overrides the last data point for today with real-time data
        {open, high, low, close, volume} with already-interpolated volume.

    Returns
    -------
    dict with keys:
        violations: list of {date, type, description, severity}
        confirmations: list of {date, type, description, severity}
        info: dict with aggregate stats
        trend_up_start: str or None
    """
    if not daily_data:
        return {'violations': [], 'confirmations': [], 'info': {}, 'trend_up_start': None}

    # If enabled_checks is None, enable all
    if enabled_checks is None:
        enabled_checks = DEFAULT_PREFERENCES

    def _enabled(key):
        return enabled_checks.get(key, True)

    # --- Shallow copy to avoid mutating cached data ---
    daily_data = list(daily_data)

    # --- Inject today's realtime data ---
    if today_realtime and daily_data:
        last = daily_data[-1]
        if last['date'] == end_date:
            # Update in place
            daily_data[-1] = {**last, **today_realtime, 'date': last['date']}
        else:
            daily_data.append({
                'date': end_date,
                'open': today_realtime.get('open', today_realtime.get('close', 0)),
                'high': today_realtime.get('high', today_realtime.get('close', 0)),
                'low': today_realtime.get('low', today_realtime.get('close', 0)),
                'close': today_realtime.get('close', 0),
                'volume': today_realtime.get('volume', 0),
                'raw_volume': today_realtime.get('raw_volume', today_realtime.get('volume', 0)),
            })

    # Safety: always compute strictly as-of end_date.
    # Some callers may pass full history to "today" even when end_date is old.
    # Truncating here guarantees trend-up, maxima, and all checks are evaluated
    # only on data available up to the selected session end.
    daily_data = [r for r in daily_data if r['date'] <= end_date]
    if not daily_data:
        return {'violations': [], 'confirmations': [], 'info': {}, 'trend_up_start': None}

    # Pre-extract arrays
    dates = [r['date'] for r in daily_data]
    opens = [r['open'] for r in daily_data]
    highs = [r['high'] for r in daily_data]
    lows = [r['low'] for r in daily_data]
    closes = [r['close'] for r in daily_data]
    volumes = [r['volume'] for r in daily_data]

    # --- Moving averages (use precomputed if available) ---
    if precomputed:
        ma10     = list(precomputed['ma10'])
        ma20     = list(precomputed['ma20'])
        ma50     = list(precomputed['ma50'])
        ma200    = list(precomputed['ma200'])
        vol_ma50 = list(precomputed['vol_ma50'])
        # Pad if daily_data grew from realtime append
        data_len = len(closes)
        for arr in (ma10, ma20, ma50, ma200, vol_ma50):
            while len(arr) < data_len:
                arr.append(arr[-1] if arr else None)
    else:
        ma10 = _sma(closes, 10)
        ma20 = _sma(closes, 20)
        ma50 = _sma(closes, 50)
        ma200 = _sma(closes, 200)
        vol_ma50 = _sma(volumes, 50)

    # --- Weekly data (always recomputed — today's bar affects current week) ---
    weekly = _aggregate_weeks(daily_data)
    weekly_vol_ma10 = _sma([w['volume'] for w in weekly], 10)

    # --- Trend-up start (use precomputed if available) ---
    if precomputed:
        trend_start_idx = precomputed['trend_start_idx']
        trend_start_date = precomputed['trend_start_date']
    else:
        trend_start_idx, trend_start_date = find_trend_up_start(dates, closes)

    # If precomputed came from a longer dataset, trend metadata may be stale
    # for this as-of-end slice. Recompute against the truncated data.
    if (
        (trend_start_idx is not None and trend_start_idx >= len(dates))
        or (trend_start_date is not None and trend_start_date > end_date)
    ):
        trend_start_idx, trend_start_date = find_trend_up_start(dates, closes)

    # --- RS series (use precomputed if available) ---
    if precomputed:
        rs_series = dict(precomputed['rs_series'])  # shallow copy — we may add today's RS
        # Recompute RS for today if realtime data was injected (closes may have changed)
        if today_realtime and sp500_data and (_enabled('rs_new_high') or _enabled('rs_new_low')):
            sp_map = {r['date']: r['close'] for r in sp500_data}
            d = dates[-1]
            sp_close = None
            # Prefer live benchmark close for current session when available;
            # this keeps RS from mixing live stock price with stale benchmark close.
            if benchmark_today_realtime and benchmark_today_realtime.get('close'):
                sp_close = benchmark_today_realtime.get('close')
            else:
                sp_close = sp_map.get(d)
            if sp_close is not None and sp_close > 0:
                rs_series[d] = closes[-1] / sp_close
    else:
        rs_series = {}
        if sp500_data and (_enabled('rs_new_high') or _enabled('rs_new_low')):
            sp_dates = [r['date'] for r in sp500_data]
            sp_closes = [r['close'] for r in sp500_data]
            rs_series = _calc_rs_series(closes, dates, sp_closes, sp_dates)

    # --- Determine analysis range indices ---
    range_start_idx = None
    range_end_idx = None
    for i, d in enumerate(dates):
        if d >= start_date and range_start_idx is None:
            range_start_idx = i
        if d <= end_date:
            range_end_idx = i

    # Clamp to nearest available trading day(s) if requested dates land on
    # non-trading days (weekends/holidays) or produce an inverted range.
    if range_start_idx is None:
        range_start_idx = len(dates) - 1 if dates else None
    if range_end_idx is None:
        range_end_idx = 0 if dates else None

    if range_start_idx is None or range_end_idx is None:
        return {'violations': [], 'confirmations': [], 'info': {},
                'trend_up_start': str(trend_start_date) if trend_start_date else None}

    if range_start_idx > range_end_idx:
        range_start_idx = range_end_idx

    range_start_date = dates[range_start_idx]
    range_end_date = dates[range_end_idx]

    violations = []
    confirmations = []
    info = {}

    def _add_v(date_val, key, desc, severity=None):
        meta = ALL_CHECKS.get(key, (desc, 'violation', 'red'))
        violations.append({
            'date': str(date_val),
            'type': key,
            'description': desc,
            'severity': severity or meta[2],
        })

    def _add_c(date_val, key, desc, severity=None):
        meta = ALL_CHECKS.get(key, (desc, 'confirmation', 'green'))
        confirmations.append({
            'date': str(date_val),
            'type': key,
            'description': desc,
            'severity': severity or meta[2],
        })

    # ===================================================================
    # PER-DAY CHECKS (iterate over each day in the monitoring range)
    # ===================================================================
    days_up = 0
    days_down = 0
    good_closes = 0
    bad_closes = 0

    # Track consecutive patterns
    megaphone_count = 0

    # For ANTS: track last 15 days
    last_15_up = []

    trend_idx = trend_start_idx if trend_start_idx is not None else 0

    # Average range for large squat/reversal
    range_pcts = []
    for i in range(range_start_idx, range_end_idx + 1):
        range_pcts.append(_day_range_pct(daily_data[i]))
    avg_range_pct = sum(range_pcts) / len(range_pcts) if range_pcts else 0

    # Build a pre-range baseline window similar to extended-warning logic:
    # start from trend-up baseline, but include ~1 year before selected start
    # when available.
    one_year_start_idx = max(0, range_start_idx - MIN_EXTENDED_LOOKBACK_DAYS)
    baseline_start_idx = min(max(trend_idx, 1), max(one_year_start_idx, 1))

    # Dynamic threshold for big up/down day — use ticker's own avg magnitude
    hist_up_pcts = []
    hist_down_pcts = []
    for hi in range(baseline_start_idx, range_start_idx):
        hpct = _pct_change(closes[hi - 1], closes[hi])
        if hpct > 0:
            hist_up_pcts.append(hpct)
        elif hpct < 0:
            hist_down_pcts.append(abs(hpct))
    avg_up_pct = (sum(hist_up_pcts) / len(hist_up_pcts)) if hist_up_pcts else 3.0
    avg_down_pct = (sum(hist_down_pcts) / len(hist_down_pcts)) if hist_down_pcts else 3.0
    big_up_threshold = max(2.0 * avg_up_pct, 3.0)
    big_down_threshold = max(2.0 * avg_down_pct, 3.0)

    # Gap alert baseline: ticker-specific 1-sigma rule from prior signed gaps.
    # Uses pre-range history to avoid look-ahead bias.
    hist_signed_gaps = [
        _pct_change(closes[gi - 1], opens[gi])
        for gi in range(baseline_start_idx, range_start_idx)
    ]
    gap_mean = (sum(hist_signed_gaps) / len(hist_signed_gaps)) if hist_signed_gaps else 0.0
    gap_std = 0.0
    if len(hist_signed_gaps) >= 2:
        gap_variance = sum((g - gap_mean) ** 2 for g in hist_signed_gaps) / len(hist_signed_gaps)
        gap_std = math.sqrt(gap_variance)

    # Daily signed change baseline as-of END date.
    # Used to ensure "largest % down day on high vol" is also statistically
    # meaningful relative to the ticker's normal day-to-day movement.
    hist_signed_day_changes = [
        _pct_change(closes[di - 1], closes[di])
        for di in range(max(trend_idx, 1), len(daily_data))
    ]
    day_change_mean = (
        sum(hist_signed_day_changes) / len(hist_signed_day_changes)
        if hist_signed_day_changes else 0.0
    )
    day_change_std = 0.0
    if len(hist_signed_day_changes) >= 2:
        day_change_variance = sum((chg - day_change_mean) ** 2 for chg in hist_signed_day_changes) / len(hist_signed_day_changes)
        day_change_std = math.sqrt(day_change_variance)
    significant_down_day_threshold = day_change_mean - day_change_std
    hist_down_day_abs = [abs(chg) for chg in hist_signed_day_changes if chg < 0]
    avg_down_day_abs = (
        sum(hist_down_day_abs) / len(hist_down_day_abs)
        if hist_down_day_abs else avg_down_pct
    )

    # Recency gate: MA-crossover and megaphone checks only count if within 2 weeks of end date.
    recency_cutoff = range_end_date - timedelta(days=14)

    for i in range(range_start_idx, range_end_idx + 1):
        row = daily_data[i]
        d = dates[i]
        prev_close = closes[i - 1] if i > 0 else closes[i]
        prev_row = daily_data[i - 1] if i > 0 else row
        day_pct = _pct_change(prev_close, closes[i])
        is_up = closes[i] > prev_close
        is_down = closes[i] < prev_close

        # --- days up / days down ---
        if is_up:
            days_up += 1
            last_15_up.append(True)
        elif is_down:
            days_down += 1
            last_15_up.append(False)
        else:
            last_15_up.append(False)
        if len(last_15_up) > 15:
            last_15_up.pop(0)

        # --- good / bad close ---
        if _close_in_upper_pct(row, 0.5):
            good_closes += 1
        else:
            bad_closes += 1

        # 1. Closes on high
        if _enabled('closes_on_high') and _close_in_upper_pct(row, 0.10):
            _add_c(d, 'closes_on_high', f'Closes on high ({closes[i]:.2f} near day high {highs[i]:.2f})')

        # 2. Closes on low
        if _enabled('closes_on_low') and _close_in_lower_pct(row, 0.10):
            _add_v(d, 'closes_on_low', f'Closes on low ({closes[i]:.2f} near day low {lows[i]:.2f})')

        # 3. Down on 50% vol increase
        if _enabled('down_50pct_vol_increase') and is_down and vol_ma50[i] is not None:
            if volumes[i] >= vol_ma50[i] * 1.5:
                if abs(day_pct) >= avg_down_day_abs:
                    vol_increase_pct = ((volumes[i] / vol_ma50[i]) - 1.0) * 100.0 if vol_ma50[i] else 0.0
                    _add_v(d, 'down_50pct_vol_increase',
                           f'Down on {vol_increase_pct:.1f}% vol increase: down {day_pct:.1f}% ({volumes[i]:,} vs 50d avg {int(vol_ma50[i]):,})')

        # 4. Up on 30%+ vol increase
        if _enabled('up_30pct_vol_increase') and is_up and vol_ma50[i] is not None:
            if volumes[i] >= vol_ma50[i] * 1.3:
                vol_increase_pct = ((volumes[i] / vol_ma50[i]) - 1.0) * 100.0 if vol_ma50[i] else 0.0
                _add_c(d, 'up_30pct_vol_increase',
                       f'Up on {vol_increase_pct:.1f}% vol increase: up {day_pct:.1f}% ({volumes[i]:,} vs 50d avg {int(vol_ma50[i]):,})')

        # 5. Bearish engulfing
        if _enabled('bearish_engulfing') and i > 0:
            if highs[i] > highs[i - 1] and closes[i] < lows[i - 1]:
                _add_v(d, 'bearish_engulfing',
                       f'Bearish engulfing (H {highs[i]:.2f}>{highs[i-1]:.2f}, C {closes[i]:.2f}<L {lows[i-1]:.2f})')

        # 6. Bullish engulfing
        if _enabled('bullish_engulfing') and i > 0:
            if lows[i] < lows[i - 1] and closes[i] > highs[i - 1]:
                _add_c(d, 'bullish_engulfing',
                       f'Bullish engulfing (L {lows[i]:.2f}<{lows[i-1]:.2f}, C {closes[i]:.2f}>H {highs[i-1]:.2f})')

        # 7. Inside day
        if _enabled('inside_day') and i > 0:
            if highs[i] < highs[i - 1] and lows[i] > lows[i - 1]:
                _add_c(d, 'inside_day', f'Inside day (H {highs[i]:.2f}<{highs[i-1]:.2f}, L {lows[i]:.2f}>{lows[i-1]:.2f})')

        # 8. MA crossover checks (only trigger on side changes from previous day)
        crossed_below_200 = (
            i > 0 and ma200[i] is not None and ma200[i - 1] is not None
            and closes[i - 1] > ma200[i - 1] and closes[i] < ma200[i]
        )
        crossed_above_200 = (
            i > 0 and ma200[i] is not None and ma200[i - 1] is not None
            and closes[i - 1] < ma200[i - 1] and closes[i] > ma200[i]
        )
        crossed_below_10 = (
            i > 0 and ma10[i] is not None and ma10[i - 1] is not None
            and closes[i - 1] > ma10[i - 1] and closes[i] < ma10[i]
        )
        crossed_below_20 = (
            i > 0 and ma20[i] is not None and ma20[i - 1] is not None
            and closes[i - 1] > ma20[i - 1] and closes[i] < ma20[i]
        )
        crossed_below_50 = (
            i > 0 and ma50[i] is not None and ma50[i - 1] is not None
            and closes[i - 1] > ma50[i - 1] and closes[i] < ma50[i]
        )
        crossed_above_50 = (
            i > 0 and ma50[i] is not None and ma50[i - 1] is not None
            and closes[i - 1] < ma50[i - 1] and closes[i] > ma50[i]
        )

        if _enabled('close_below_200ma') and crossed_below_200:
            _add_v(d, 'close_below_200ma', f'Close {closes[i]:.2f} crossed below 200MA {ma200[i]:.2f}')
        if _enabled('close_above_200ma') and crossed_above_200:
            _add_c(d, 'close_above_200ma', f'Close {closes[i]:.2f} crossed above 200MA {ma200[i]:.2f}')
        if _enabled('close_below_10ma') and crossed_below_10 and d >= recency_cutoff:
            _add_v(d, 'close_below_10ma', f'Close {closes[i]:.2f} crossed below 10MA {ma10[i]:.2f}')
        if _enabled('close_below_20ma') and crossed_below_20 and d >= recency_cutoff:
            _add_v(d, 'close_below_20ma', f'Close {closes[i]:.2f} crossed below 20MA {ma20[i]:.2f}')
        if _enabled('close_below_50ma') and crossed_below_50 and d >= recency_cutoff:
            _add_v(d, 'close_below_50ma', f'Close {closes[i]:.2f} crossed below 50MA {ma50[i]:.2f}')
        if _enabled('close_above_50ma') and crossed_above_50 and d >= recency_cutoff:
            _add_c(d, 'close_above_50ma', f'Close {closes[i]:.2f} crossed above 50MA {ma50[i]:.2f}')
        if _enabled('close_below_50ma_high_vol') and crossed_below_50 and d >= recency_cutoff and vol_ma50[i] is not None:
            if volumes[i] >= vol_ma50[i] * 1.5:
                _add_v(d, 'close_below_50ma_high_vol',
                       f'Close crossed below 50MA on high vol ({volumes[i]:,} vs 50d avg {int(vol_ma50[i]):,})')

        # 9. Big down/up day (dynamic threshold based on ticker avg)
        if _enabled('big_down_day') and day_pct <= -big_down_threshold:
            _add_v(d, 'big_down_day', f'Down {day_pct:.1f}%')
        if _enabled('big_up_day') and day_pct >= big_up_threshold:
            _add_c(d, 'big_up_day', f'Up +{day_pct:.1f}%')

        # 10. Gap alert (ticker-specific 2-sigma move in signed gap)
        if _enabled('gap_alert') and i > 0:
            signed_gap_pct = _pct_change(closes[i - 1], opens[i])
            threshold_sigma = 2.0
            threshold = gap_std * threshold_sigma
            if gap_std > 0 and abs(signed_gap_pct - gap_mean) >= threshold:
                direction = 'up' if signed_gap_pct >= 0 else 'down'
                _add_v(
                    d,
                    'gap_alert',
                    f'Gap {direction} {abs(signed_gap_pct):.1f}% (outside 2σ: mean {gap_mean:.2f}%, σ {gap_std:.2f}%)',
                    'red',
                )

        # 11. 20%+ above 20-day MA
        if _enabled('above_20day_20pct') and i > 0 and ma20[i] is not None and ma20[i - 1] is not None:
            above_pct = _pct_change(ma20[i], closes[i])
            prev_above_pct = _pct_change(ma20[i - 1], closes[i - 1])
            if above_pct >= 20.0 and prev_above_pct < 20.0:
                _add_v(d, 'above_20day_20pct', f'{above_pct:.1f}% above 20-day MA: close {closes[i]:.2f} vs 20MA {ma20[i]:.2f}', 'red')

        # 12. Angle D warning (trigger only when 50MA first turns down/flat)
        if _enabled('angle_d_warning') and i >= 4 and ma50[i] is not None and ma50[i - 4] is not None:
            ma50_change = _pct_change(ma50[i - 4], ma50[i])
            prev_ma50_change = None
            if i >= 5 and ma50[i - 1] is not None and ma50[i - 5] is not None:
                prev_ma50_change = _pct_change(ma50[i - 5], ma50[i - 1])

            started_downtrend = ma50_change <= 0 and (prev_ma50_change is None or prev_ma50_change > 0)
            # If monitoring starts while 50MA is already down/flat, still emit once
            # on the first day in range so the condition is visible to the user.
            first_range_day_already_downtrend = (i == range_start_idx and ma50_change <= 0)

            should_emit_angle_d = started_downtrend or first_range_day_already_downtrend
            if should_emit_angle_d and d >= recency_cutoff:
                    _add_v(d, 'angle_d_warning',
                        'Angle D Warning (50MA Turned Down/Flat)', 'red')

        # 13. Large squat/reversal
        if _enabled('large_squat_reversal') and avg_range_pct > 0:
            curr_range = _day_range_pct(row)
            if curr_range >= 2 * avg_range_pct:
                # Require a bearish close in the lower half of the day's range.
                # This avoids flagging large-range up days as reversals.
                mid = (row['high'] + row['low']) / 2
                if is_down and closes[i] <= mid:
                    _add_v(
                        d,
                        'large_squat_reversal',
                        f'Large reversal: range {curr_range:.1f}% is 2x avg {avg_range_pct:.1f}%, close {closes[i]:.2f} <= mid {mid:.2f}'
                    )

        # --- Megaphone check (engulfing = high > prev high AND low < prev low) ---
        if i > 0:
            is_engulfing_day = highs[i] > highs[i - 1] and lows[i] < lows[i - 1]
            if is_engulfing_day:
                megaphone_count += 1
            else:
                megaphone_count = 0

            if _enabled('daily_megaphone') and megaphone_count >= 2 and d >= recency_cutoff:
                _add_v(d, 'daily_megaphone', f'{megaphone_count} consecutive megaphone/engulfing days')

        # --- ANTS check (12 of 15 days up + volume/gain quality filters) ---
        if _enabled('ants') and len(last_15_up) >= 15:
            up_count = sum(1 for x in last_15_up if x)
            if up_count >= 12:
                window_start = i - 14
                window_volumes = volumes[window_start:i + 1]
                window_ma50 = vol_ma50[window_start:i + 1]

                # Require a usable 50d MA for the full 15-day window.
                if all(v is not None for v in window_ma50):
                    avg_window_volume = sum(window_volumes) / len(window_volumes)
                    avg_window_ma50 = sum(window_ma50) / len(window_ma50)

                    volume_strength_ok = avg_window_volume >= (avg_window_ma50 * 1.2)
                    gain_15d = _pct_change(closes[window_start], closes[i])
                    gain_strength_ok = gain_15d >= 20.0

                    if volume_strength_ok and gain_strength_ok:
                        _add_c(
                            d,
                            'ants',
                            f'ANTS: {up_count} of last 15 days up, 15d gain {gain_15d:.1f}%, avg vol {int(avg_window_volume):,} vs 50d avg {int(avg_window_ma50):,}'
                        )



    # ===================================================================
    # PROGRESSIVE SINCE-TREND-UP RECORDS (daily)
    # Emits each time a new record is set, but only if that record day
    # falls inside the selected start/end view — same pattern as
    # widest_spread and the weekly record checks.
    # ===================================================================
    if trend_start_idx is not None:
        _trend_vol_record = None
        _trend_down_pct_record = None   # stored as positive magnitude
        _trend_down_pct_hv_record = None
        _trend_gap_record = None

        for i in range(max(trend_start_idx, 1), len(daily_data)):
            d_i = dates[i]
            prev_c = closes[i - 1]
            day_p = _pct_change(prev_c, closes[i])
            is_down_i = closes[i] < prev_c
            is_up_i = closes[i] > prev_c
            vol_i = volumes[i]
            in_view = range_start_date <= d_i <= range_end_date

            # -- down_up_largest_vol (down = violation, up = confirmation) --
            if _enabled('down_up_largest_vol'):
                if _trend_vol_record is None or vol_i > _trend_vol_record:
                    _trend_vol_record = vol_i
                    if in_view:
                        if is_down_i:
                            _add_v(d_i, 'down_up_largest_vol',
                                   f'Down on new largest vol since trend up ({vol_i:,})')
                        elif is_up_i:
                            _add_c(d_i, 'down_up_largest_vol',
                                   f'Up on new largest vol since trend up ({vol_i:,})', 'green')

            # -- largest_down_day_pct --
            if _enabled('largest_down_day_pct') and is_down_i:
                abs_p = abs(day_p)
                if _trend_down_pct_record is None or abs_p > _trend_down_pct_record:
                    _trend_down_pct_record = abs_p
                    if in_view:
                        _add_v(d_i, 'largest_down_day_pct',
                               f'New largest down day ({day_p:.1f}%) since trend up')

            # -- largest_pct_down_high_vol --
            if _enabled('largest_pct_down_high_vol') and is_down_i:
                abs_p = abs(day_p)
                if vol_ma50[i] is not None and vol_i >= vol_ma50[i] * 1.5:
                    is_significant_down_day = day_p <= significant_down_day_threshold
                    if is_significant_down_day and (_trend_down_pct_hv_record is None or abs_p > _trend_down_pct_hv_record):
                        _trend_down_pct_hv_record = abs_p
                        if in_view:
                            _add_v(d_i, 'largest_pct_down_high_vol',
                                   f'New largest % down day ({day_p:.1f}%) on high vol since trend up')

            # -- largest_gap --
            if _enabled('largest_gap'):
                gap_i = abs(_pct_change(prev_c, opens[i]))
                if gap_i > 0 and (_trend_gap_record is None or gap_i > _trend_gap_record):
                    _trend_gap_record = gap_i
                    if in_view:
                        _add_v(d_i, 'largest_gap',
                               f'New largest gap ({gap_i:.1f}%) since trend up')

    # ===================================================================
    # DAILY HIGHER HIGHS / LOWER LOWS (from current day backward)
    # ===================================================================
    if range_start_idx is not None and range_end_idx is not None and range_end_idx >= range_start_idx:
        streak_end_idx = range_end_idx

        # Count consecutive daily higher highs backward from current day.
        # Example: today>yesterday>day-2 then stop -> count=3.
        hh_streak = 1
        idx = streak_end_idx
        while idx > range_start_idx and highs[idx] > highs[idx - 1]:
            hh_streak += 1
            idx -= 1

        # Count consecutive daily lower lows backward from current day.
        ll_streak = 1
        idx = streak_end_idx
        while idx > range_start_idx and lows[idx] < lows[idx - 1]:
            ll_streak += 1
            idx -= 1

        current_day = dates[streak_end_idx]
        if _enabled('daily_higher_highs') and hh_streak >= 3:
            _add_c(current_day, 'daily_higher_highs', f'{hh_streak} consecutive daily higher highs')
        if _enabled('daily_lower_lows') and ll_streak >= 3:
            _add_v(current_day, 'daily_lower_lows', f'{ll_streak} consecutive daily lower lows')

    # ===================================================================
    # SQUAT/REVERSAL ON BUY DAY
    # ===================================================================
    if _enabled('squat_reversal') and range_start_idx is not None:
        buy_row = daily_data[range_start_idx]
        rng = buy_row['high'] - buy_row['low']
        if rng > 0:
            mid = buy_row['low'] + rng * 0.5
            if buy_row['close'] <= mid:
                  _add_v(range_start_date, 'squat_reversal',
                     f'Squat on buy day: close {buy_row["close"]:.2f} in lower 50% of range ({buy_row["low"]:.2f}-{buy_row["high"]:.2f})')

    # ===================================================================
    # RS NEW HIGH / LOW (52-week, matching MarketSurge RS Line behavior)
    # RS Line = stock_price / SP500_price
    # ===================================================================
    if _enabled('rs_new_high') or _enabled('rs_new_low'):
        if rs_series:
            # Compare each day against the prior 252 RS observations (about 52 weeks).
            # This is robust to missing dates and aligns with trading-session windows.
            rs_lookback_days = 252
            sorted_rs_dates = sorted(rs_series.keys())
            for idx, d in enumerate(sorted_rs_dates):
                if d < range_start_date or d > range_end_date:
                    continue
                if idx == 0:
                    continue

                v = rs_series[d]
                window_start_idx = max(0, idx - rs_lookback_days)
                window_dates = sorted_rs_dates[window_start_idx:idx]
                window_vals = [rs_series[wd] for wd in window_dates]
                if not window_vals:
                    continue

                window_max = max(window_vals)
                window_min = min(window_vals)
                if _enabled('rs_new_high') and v > window_max:
                    _add_c(d, 'rs_new_high', f'RS Line at new 52-week high')
                if _enabled('rs_new_low') and v < window_min:
                    _add_v(d, 'rs_new_low', f'RS Line at new 52-week low')

    # ===================================================================
    # WEEKLY CHECKS
    # ===================================================================
    # Filter weeks that overlap the selected analysis range.
    range_weeks = [
        w for w in weekly
        if w.get('end_date', w['date']) >= range_start_date and w['date'] <= range_end_date
    ]

    # Weekly lower lows / higher highs streak from current week backward
    if len(range_weeks) >= 2:
        weekly_hh_streak = 1
        wi = len(range_weeks) - 1
        while wi > 0 and range_weeks[wi]['high'] > range_weeks[wi - 1]['high']:
            weekly_hh_streak += 1
            wi -= 1

        weekly_ll_streak = 1
        wi = len(range_weeks) - 1
        while wi > 0 and range_weeks[wi]['low'] < range_weeks[wi - 1]['low']:
            weekly_ll_streak += 1
            wi -= 1

        current_week_date = range_weeks[-1].get('end_date', range_weeks[-1]['date'])
        if _enabled('weekly_higher_highs') and weekly_hh_streak >= 3:
            _add_c(current_week_date, 'weekly_higher_highs',
                   f'{weekly_hh_streak} consecutive weekly higher highs')
        if _enabled('weekly_lower_lows') and weekly_ll_streak >= 3:
            _add_v(current_week_date, 'weekly_lower_lows',
                   f'{weekly_ll_streak} consecutive weekly lower lows')

    # Largest down week on high vol / largest % down week across the trend-up window.
    # Emit each new record-setting worst down week, but only if that record week
    # falls inside the selected start/end view.
    if trend_start_date is not None and len(weekly) >= 2:
        worst_down_pct_so_far = None
        for wi in range(1, len(weekly)):
            current_week = weekly[wi]
            if current_week.get('end_date', current_week['date']) < trend_start_date:
                continue
            wpct = _pct_change(weekly[wi - 1]['close'], current_week['close'])
            if wpct >= 0:
                continue

            is_new_record = worst_down_pct_so_far is None or wpct < worst_down_pct_so_far
            if not is_new_record:
                continue

            worst_down_pct_so_far = wpct

            if current_week.get('end_date', current_week['date']) < range_start_date or current_week['date'] > range_end_date:
                continue

            week_emit_date = current_week.get('end_date', current_week['date'])
            if _enabled('largest_pct_down_week'):
                _add_v(week_emit_date, 'largest_pct_down_week',
                       f'New largest % down week ({wpct:.1f}%) since trend up')

            if _enabled('largest_down_week_high_vol') and weekly_vol_ma10 and wi < len(weekly_vol_ma10):
                wvma = weekly_vol_ma10[wi]
                if wvma is not None and current_week['volume'] >= wvma * 1.5:
                    _add_v(week_emit_date, 'largest_down_week_high_vol',
                           f'New largest down week ({wpct:.1f}%) on high vol ({current_week["volume"]:,} vs 10w avg {int(wvma):,}) since trend up')

    # Widest spread since trend up.
    # Emit each new record-setting widest spread day, but only when that
    # record day falls inside the selected start/end view.
    if _enabled('widest_spread') and trend_start_idx is not None:
        widest_spread_so_far = None
        for i in range(trend_start_idx, len(daily_data)):
            spread = _day_range_pct(daily_data[i])
            if spread <= 0:
                continue

            is_new_record = widest_spread_so_far is None or spread > widest_spread_so_far
            if not is_new_record:
                continue

            widest_spread_so_far = spread
            if dates[i] < range_start_date or dates[i] > range_end_date:
                continue

            _add_v(dates[i], 'widest_spread',
                   f'New widest spread ({spread:.1f}%) since trend up')

    # ===================================================================
    # EXTENDED / VERY EXTENDED WARNING
    # Percentile model based on the stock's own historical rallies.
    # Baseline starts from trend-up start, but extends backward to include
    # at least ~1 trading year before move start when available.
    # Evaluate each period from buy/start date -> each day in range,
    # and only emit on state transitions.
    # ===================================================================
    if (_enabled('extended_warning') or _enabled('very_extended')) and trend_start_idx is not None:
        if range_start_idx is not None and range_end_idx is not None:
            prior_rallies_by_duration = {}

            def _get_prior_rallies_for_duration(duration):
                cached = prior_rallies_by_duration.get(duration)
                if cached is not None:
                    return cached

                if duration <= 0:
                    prior_rallies_by_duration[duration] = []
                    return []

                historical_end_limit = range_start_idx - 1
                one_year_start_idx = max(0, range_start_idx - MIN_EXTENDED_LOOKBACK_DAYS)
                history_start_idx = min(trend_start_idx, one_year_start_idx)
                max_start_i = historical_end_limit - duration
                if max_start_i < history_start_idx:
                    prior_rallies_by_duration[duration] = []
                    return []

                rallies = []
                for start_i in range(history_start_idx, max_start_i + 1):
                    end_i = start_i + duration
                    rallies.append(_pct_change(closes[start_i], closes[end_i]))

                rallies.sort()
                prior_rallies_by_duration[duration] = rallies
                return rallies

            # States for transition-based emission:
            # none -> extended -> very
            previous_state = 'none'

            for end_i in range(range_start_idx + 1, range_end_idx + 1):
                rally_duration = end_i - range_start_idx
                current_rally_pct = _pct_change(closes[range_start_idx], closes[end_i])
                sorted_rallies = _get_prior_rallies_for_duration(rally_duration)
                if not sorted_rallies:
                    previous_state = 'none'
                    continue

                sample_size = len(sorted_rallies)
                extended_label = int(round(EXTENDED_WARNING_PERCENTILE * 100))
                very_label = int(round(VERY_EXTENDED_PERCENTILE * 100))
                p_extended_idx = min(sample_size - 1, max(0, math.ceil(sample_size * EXTENDED_WARNING_PERCENTILE) - 1))
                p_very_idx = min(sample_size - 1, max(0, math.ceil(sample_size * VERY_EXTENDED_PERCENTILE) - 1))
                p_extended_level = sorted_rallies[p_extended_idx]
                p_very_level = sorted_rallies[p_very_idx]
                current_pct_rank = _percentile_rank(sorted_rallies, current_rally_pct)

                current_state = 'none'
                if current_rally_pct >= p_very_level:
                    current_state = 'very'
                elif current_rally_pct >= p_extended_level:
                    current_state = 'extended'

                current_date = dates[end_i]

                # Emit only on entry into very-extended from non-very states.
                if current_state == 'very' and previous_state != 'very':
                    if _enabled('very_extended'):
                        _add_v(
                            current_date,
                            'very_extended',
                            f'Very extended: {rally_duration}-day rally {current_rally_pct:.1f}% is at the {current_pct_rank:.0f}th percentile vs {sample_size} prior ticker-specific rallies ({very_label}th pct threshold {p_very_level:.1f}%)'
                        )

                # Emit extended warning only on entry from non-extended state.
                # This avoids repeated daily alerts and excludes very->extended downgrades.
                elif current_state == 'extended' and previous_state == 'none':
                    if _enabled('extended_warning'):
                        _add_v(
                            current_date,
                            'extended_warning',
                            f'Extended warning: {rally_duration}-day rally {current_rally_pct:.1f}% is at the {current_pct_rank:.0f}th percentile vs {sample_size} prior ticker-specific rallies ({extended_label}th pct threshold {p_extended_level:.1f}%)',
                            'red'
                        )

                previous_state = current_state

    # ===================================================================
    # DAYS UP/DOWN RATIO (info, always shown, colored conditionally)
    # ===================================================================
    min_ratio_sample_days = 5
    monitored_trading_days = (range_end_idx - range_start_idx + 1) if (
        range_start_idx is not None and range_end_idx is not None
    ) else 0
    has_min_ratio_sample = monitored_trading_days >= min_ratio_sample_days

    total_days = days_up + days_down
    days_up_pct = (days_up / total_days * 100) if total_days > 0 else 50
    days_up_color = (
        'green' if has_min_ratio_sample and days_up_pct >= 70
        else ('red' if has_min_ratio_sample and days_up_pct <= 30 else 'gray')
    )

    if has_min_ratio_sample and days_up_pct >= 70:
        _add_c(
            range_end_date,
            'days_up_down',
            f'Days up/down: {days_up}/{days_down} ({days_up_pct:.0f}% up)',
            'green'
        )
    elif has_min_ratio_sample and days_up_pct <= 30:
        _add_v(
            range_end_date,
            'days_up_down',
            f'Days up/down: {days_up}/{days_down} ({days_up_pct:.0f}% up)',
            'red'
        )

    info['days_up'] = days_up
    info['days_down'] = days_down
    info['days_up_pct'] = round(days_up_pct, 1)
    info['days_up_color'] = days_up_color

    # ===================================================================
    # GOOD/BAD CLOSE RATIO (info, always shown, colored conditionally)
    # ===================================================================
    total_closes = good_closes + bad_closes
    good_close_pct = (good_closes / total_closes * 100) if total_closes > 0 else 50
    close_color = (
        'green' if has_min_ratio_sample and good_close_pct >= 70
        else ('red' if has_min_ratio_sample and good_close_pct <= 30 else 'gray')
    )

    if has_min_ratio_sample and good_close_pct >= 70:
        _add_c(
            range_end_date,
            'good_bad_close',
            f'Good/bad closes: {good_closes}/{bad_closes} ({good_close_pct:.0f}% good)',
            'green'
        )
    elif has_min_ratio_sample and good_close_pct <= 30:
        _add_v(
            range_end_date,
            'good_bad_close',
            f'Good/bad closes: {good_closes}/{bad_closes} ({good_close_pct:.0f}% good)',
            'red'
        )

    info['good_closes'] = good_closes
    info['bad_closes'] = bad_closes
    info['good_close_pct'] = round(good_close_pct, 1)
    info['good_close_color'] = close_color

    current_idx = range_end_idx
    if current_idx is not None and closes:
        curr_close = closes[current_idx]
        prev_close = closes[current_idx - 1] if current_idx > 0 else curr_close

        info['current_price'] = round(curr_close, 2)
        info['current_change_pct'] = round(((curr_close - prev_close) / prev_close * 100), 2) if prev_close else 0

    return {
        'violations': violations,
        'confirmations': confirmations,
        'info': info,
        'risk_analysis': (precomputed.get('risk_analysis') if precomputed else compute_risk_analysis(daily_data, lookback_days=252)),
        'trend_up_start': str(trend_start_date) if trend_start_date else None,
    }
