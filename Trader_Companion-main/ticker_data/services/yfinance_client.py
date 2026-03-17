import time
import logging
import yfinance as yf
from .base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

class YFinanceClient(BaseFetcher):
    PROVIDER_NAME = "yfinance"

    def fetch_data(self, symbol: str, force_full: bool = False, period_override: str | None = None):
        try:
            started = time.perf_counter()
            request_count = 0
            net_ms_total = 0.0

            self._enforce_rate_limit(consumed_requests=1)
            request_count += 1
            
            ticker = yf.Ticker(symbol)
            
            # Fetch daily data
            period_daily = '5d'
            if force_full:
                period_daily = 'max' if symbol.upper() == 'SPY' else '10y'
            if period_override and not (force_full and symbol.upper() == 'SPY'):
                period_daily = period_override
            _t0 = time.perf_counter()
            df_daily = ticker.history(period=period_daily, interval='1d')
            net_ms_total += (time.perf_counter() - _t0) * 1000.0
            
            records_daily = []
            if not df_daily.empty:
                for idx, row in df_daily.iterrows():
                    d = idx.date() if hasattr(idx, 'date') else idx
                    d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10]
                    records_daily.append({
                        'date': d_str,
                        'open': round(float(row.get('Open', 0)), 4),
                        'high': round(float(row.get('High', 0)), 4),
                        'low': round(float(row.get('Low', 0)), 4),
                        'close': round(float(row.get('Close', 0)), 4),
                        'volume': int(row.get('Volume', 0))
                    })
                self._save_daily_bars(symbol, records_daily)

            # Weekly bars are only fetched for SPY during initial/full historical backfill.
            if force_full and symbol.upper() == 'SPY':
                time.sleep(0.1)
                self._enforce_rate_limit(consumed_requests=1)
                request_count += 1
                _t0 = time.perf_counter()
                df_weekly = ticker.history(period='max', interval='1wk')
                net_ms_total += (time.perf_counter() - _t0) * 1000.0

                records_weekly = []
                if not df_weekly.empty:
                    for idx, row in df_weekly.iterrows():
                        d = idx.date() if hasattr(idx, 'date') else idx
                        d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10]
                        records_weekly.append({
                            'date': d_str,
                            'open': round(float(row.get('Open', 0)), 4),
                            'high': round(float(row.get('High', 0)), 4),
                            'low': round(float(row.get('Low', 0)), 4),
                            'close': round(float(row.get('Close', 0)), 4),
                            'volume': int(row.get('Volume', 0))
                        })
                    self._save_weekly_bars(symbol, records_weekly)
            

            net_ms = round(net_ms_total, 2)
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            details = f"1d {period_daily}"
            if force_full and symbol.upper() == 'SPY':
                details += ", 1w max"
            self._log_request(symbol, success=True, net_ms=net_ms, duration_ms=duration_ms, count=request_count, details=details)
            return True

        except Exception as e:
            logger.error(f"[YFinance] Failed fetching {symbol}: {e}")
            self._log_request(symbol, success=False, net_ms=0.0, duration_ms=0.0, count=1, details="Failed")
            return False
