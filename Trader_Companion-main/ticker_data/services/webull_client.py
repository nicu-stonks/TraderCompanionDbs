import os
import time
import json
import logging
import threading
from datetime import datetime
import pandas as pd
from typing import Optional
from webull import webull
from django.conf import settings
from .base_fetcher import BaseFetcher
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

_WEBULL_SINGLETON = None
_WEBULL_SINGLETON_LOCK = threading.Lock()


def get_shared_webull_client():
    """Return the process-wide WebullClient singleton."""
    global _WEBULL_SINGLETON
    if _WEBULL_SINGLETON is None:
        with _WEBULL_SINGLETON_LOCK:
            if _WEBULL_SINGLETON is None:
                _WEBULL_SINGLETON = WebullClient()
    return _WEBULL_SINGLETON


class WebullClient(BaseFetcher):
    PROVIDER_NAME = "webull"

    def __init__(self):
        super().__init__()
        self.wb = webull()
        self._status = 'disconnected'
        self._login_thread = None
        self._last_error = ''
        self._login_in_progress = False
        self._request_count = 0
        self._health_check_interval = 60
        self._last_health_check_ts = 0.0
        self._consecutive_health_failures = 0
        self.credentials_file = os.path.join(settings.BASE_DIR, 'dbs', 'webull_credentials.json')
        self.load_session()

    def get_status(self):
        # Trigger a health check so the UI always reflects the real state
        if self._status == 'connected' and not self._login_in_progress:
            self._maybe_health_check()

        has_session = os.path.exists(self.credentials_file)
        normalized_status = self._status
        if normalized_status == 'connecting':
            normalized_status = 'login_required'

        return {
            'status': normalized_status,
            'login_in_progress': self._login_in_progress,
            'is_logging_in': self._login_in_progress,
            'last_error': self._last_error or None,
            'error': self._last_error,
            'request_count': self._request_count,
            'has_session': has_session,
        }

    def start_login(self):
        if self._login_in_progress:
            return {'status': 'Already logging in'}
            
        self._login_in_progress = True
        self._status = 'connecting'
        self._last_error = ''
        
        self._login_thread = threading.Thread(target=self._run_inline_login)
        self._login_thread.daemon = True
        self._login_thread.start()
        
        return {'status': 'Login started'}

    def _run_inline_login(self):
        driver = None
        try:
            logger.info("Initializing Selenium for Webull login...")
            chrome_options = Options()
            # Interactive login must be visible for manual auth/2FA.
            chrome_options.add_experimental_option("detach", False)
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Need a user agent to avoid headless blocking
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            chrome_options.add_argument(f"user-agent={user_agent}")

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            logger.info("Opening Webull login page for manual authentication...")
            driver.get('https://app.webull.com/trade')

            timeout_seconds = 240
            poll_interval_seconds = 1
            deadline = time.time() + timeout_seconds
            tokens = None

            logger.info(
                "Waiting up to %ss for Webull login completion and token availability...",
                timeout_seconds,
            )

            while time.time() < deadline:
                access_token = None
                did = None
                uuid = None
                refresh_token = ''
                trade_token = ''
                device_id = ''

                try:
                    cookies = driver.get_cookies()
                    cookie_dict = {c.get('name'): c.get('value') for c in cookies}
                    access_token = cookie_dict.get('web_lt')
                    did = cookie_dict.get('web_did')
                    uuid = cookie_dict.get('web_uid')
                except Exception:
                    pass

                if not access_token or not uuid:
                    try:
                        logs = driver.execute_script("return window.localStorage;") or {}
                        access_token = access_token or logs.get('access_token') or logs.get('web_lt')
                        uuid = uuid or logs.get('uuid') or logs.get('web_uid')
                        refresh_token = logs.get('refresh_token') or logs.get('web_rt') or ''
                        trade_token = logs.get('trade_token', '')
                        device_id = logs.get('device_id', '')
                        did = did or device_id
                    except Exception:
                        pass

                if access_token and (did or device_id) and uuid:
                    tokens = {
                        'uuid': uuid,
                        'access_token': access_token,
                        'refresh_token': refresh_token,
                        'trade_token': trade_token,
                        'device_id': device_id or did,
                        'did': did or device_id,
                        'saved_at': datetime.utcnow().isoformat() + 'Z',
                    }
                    break

                time.sleep(poll_interval_seconds)
            
            if tokens and tokens.get('access_token'):
                with open(self.credentials_file, 'w') as f:
                    json.dump(tokens, f)
                logger.info("Successfully acquired Webull tokens via inline Selenium.")
                self.load_session()
            else:
                self._status = 'disconnected'
                self._last_error = 'Login timeout or failed to extract tokens from browser.'
                logger.error("Failed to extract tokens via Selenium before timeout.")
                
        except Exception as e:
            self._status = 'disconnected'
            self._last_error = str(e)
            logger.error(f"Inline Webull login failed: {e}")
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass
            self._login_in_progress = False

    def load_session(self):
        if not os.path.exists(self.credentials_file):
            self._status = 'login_required'
            return False
            
        try:
            with open(self.credentials_file, 'r') as f:
                creds = json.load(f)
            
            did = creds.get('did') or creds.get('device_id', '')

            self.wb._device_id = creds.get('device_id', '') or did
            self.wb._did = did
            self.wb._access_token = creds.get('access_token', '')
            self.wb._refresh_token = creds.get('refresh_token', '')
            self.wb._uuid = creds.get('uuid', '')
            self.wb._trade_token = creds.get('trade_token', '')
            
            if not self.wb._access_token or not self.wb._uuid:
                self._status = 'login_required'
                return False
                
            resp = self.wb.get_account_id()
            if isinstance(resp, dict) and 'success' in resp and not resp['success']:
                self._status = 'login_required'
                return False
                
            self._status = 'connected'
            self._last_error = ''
            self._consecutive_health_failures = 0
            return True
            
        except Exception as e:
            self._status = 'login_required'
            logger.error(f"[WebullClient] Failed to load session: {e}")
            return False

    def _maybe_health_check(self):
        """Periodically validate Webull session and downgrade status on failures."""
        if self._status != 'connected' or not self.wb:
            return self._status == 'connected'

        now = time.time()
        if (now - self._last_health_check_ts) < self._health_check_interval:
            return True

        self._last_health_check_ts = now

        try:
            quote = self.wb.get_quote(stock='AAPL')
            if quote and (quote.get('close') or quote.get('price')):
                self._consecutive_health_failures = 0
                self._last_error = ''
                return True

            self._status = 'login_required'
            self._last_error = 'Session may have expired (health check failed)'
            self._consecutive_health_failures = 0
            logger.warning("[WebullClient] Health check failed - no data returned")
            return False
        except Exception as e:
            error_lower = str(e).lower()
            if 'token' in error_lower or 'unauthorized' in error_lower or 'illegal' in error_lower:
                self._status = 'login_required'
                self._last_error = 'Session expired (auth error)'
                self._consecutive_health_failures = 0
                logger.warning("[WebullClient] Health check auth error: %s", e)
                return False

            self._consecutive_health_failures += 1
            logger.warning(
                "[WebullClient] Health check network error (%s/3): %s",
                self._consecutive_health_failures,
                e,
            )
            if self._consecutive_health_failures >= 3:
                self._status = 'disconnected'
                self._last_error = 'Cannot reach Webull API - check network or retry login'
                return False
            return True

    def fetch_data(self, symbol: str, force_full: bool = False, count_override: int | None = None, period_override: str | None = None):
        if self._status != 'connected' or not self.wb:
            logger.error(f"[WebullClient] Not connected. Cannot fetch {symbol}.")
            return False

        if not self._maybe_health_check():
            logger.warning("[WebullClient] Skipping fetch for %s due to failed health check.", symbol)
            return False

        try:
            started = time.perf_counter()
            requests_made = 0
            net_ms_total = 0.0

            # Daily bars — Webull hard-caps at 1200 bars per request
            count_daily = 1200 if force_full else 10
            if count_override is not None:
                count_daily = count_override
            count_daily = min(count_daily, 1200)
            self._enforce_rate_limit(consumed_requests=1)
            requests_made += 1
            self._request_count += 1
            _t0 = time.perf_counter()
            daily_bars = self.wb.get_bars(stock=symbol, interval='d1', count=count_daily)
            net_ms_total += (time.perf_counter() - _t0) * 1000.0

            records_daily = []
            if isinstance(daily_bars, pd.DataFrame) and not daily_bars.empty:
                for idx, row in daily_bars.iterrows():
                    d = idx.date() if hasattr(idx, 'date') else idx
                    d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10]
                    row_dict = {k.lower(): v for k, v in row.items()}
                    records_daily.append({
                        'date': d_str,
                        'open': round(float(row_dict.get('open', 0)), 4),
                        'high': round(float(row_dict.get('high', 0)), 4),
                        'low': round(float(row_dict.get('low', 0)), 4),
                        'close': round(float(row_dict.get('close', row_dict.get('price', 0))), 4),
                        'volume': int(row_dict.get('volume', 0)),
                    })
                self._save_daily_bars(symbol, records_daily)

            # Weekly bars are only fetched for SPY during initial/full backfill.
            if force_full and symbol.upper() == 'SPY':
                time.sleep(0.1)
                self._enforce_rate_limit(consumed_requests=1)
                requests_made += 1
                self._request_count += 1
                try:
                    _t0 = time.perf_counter()
                    weekly_bars = self.wb.get_bars(stock=symbol, interval='w1', count=1200)
                    net_ms_total += (time.perf_counter() - _t0) * 1000.0
                except Exception as weekly_err:
                    logger.warning("[WebullClient] Weekly fetch failed for %s: %s", symbol, weekly_err)
                    weekly_bars = None

                records_weekly = []
                if isinstance(weekly_bars, pd.DataFrame) and not weekly_bars.empty:
                    for idx, row in weekly_bars.iterrows():
                        d = idx.date() if hasattr(idx, 'date') else idx
                        d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10]
                        row_dict = {k.lower(): v for k, v in row.items()}
                        records_weekly.append({
                            'date': d_str,
                            'open': round(float(row_dict.get('open', 0)), 4),
                            'high': round(float(row_dict.get('high', 0)), 4),
                            'low': round(float(row_dict.get('low', 0)), 4),
                            'close': round(float(row_dict.get('close', row_dict.get('price', 0))), 4),
                            'volume': int(row_dict.get('volume', 0)),
                        })
                    self._save_weekly_bars(symbol, records_weekly)

            net_ms = round(net_ms_total, 2)
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            details = f"1d {'1200' if force_full else '10'} bars"
            if force_full and symbol.upper() == 'SPY':
                details += ", 1w max"
            self._log_request(symbol, success=True, net_ms=net_ms, duration_ms=duration_ms, count=max(1, requests_made), details=details)
            return True

        except Exception as e:
            logger.error(f"[WebullClient] Failed fetching {symbol}: {e}")
            error_lower = str(e).lower()
            if 'token' in error_lower or 'unauthorized' in error_lower or 'illegal' in error_lower:
                self._status = 'login_required'
                self._last_error = 'Session expired - please re-login'
            self._log_request(symbol, success=False, net_ms=0.0, duration_ms=0.0, count=1, details="Failed")
            return False
