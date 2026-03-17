#!/usr/bin/env python3
import pytz
from datetime import datetime, timedelta, time as dt_time
import requests
import time
import argparse
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
from typing import List, Dict, Optional, Tuple
import random
import os
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def manage_log_files(max_files=20):
    """Manage log files with round-robin rotation"""
    log_pattern = "stock_data_server_*.log"
    existing_log_files = glob.glob(log_pattern)
    
    if not existing_log_files:
        return 1
    
    # Extract numbers and sort by modification time (oldest first)
    log_files_with_info = []
    for file in existing_log_files:
        try:
            num = int(file.split("_")[-1].split(".")[0])
            mtime = os.path.getmtime(file)
            log_files_with_info.append((file, num, mtime))
        except (ValueError, IndexError, OSError):
            continue
    
    # Sort by modification time (oldest first)
    log_files_with_info.sort(key=lambda x: x[2])
    
    # If we have max_files or more, delete the oldest ones
    while len(log_files_with_info) >= max_files:
        oldest_file = log_files_with_info.pop(0)[0]
        try:
            os.remove(oldest_file)
            logger.info(f"Deleted old log file: {oldest_file}")
        except OSError as e:
            logger.warning(f"Could not delete {oldest_file}: {e}")
    
    # Find next available number
    existing_numbers = [info[1] for info in log_files_with_info]
    if existing_numbers:
        next_num = max(existing_numbers) + 1
    else:
        next_num = 1
    
    return next_num

# Use the function to get the next log file number
next_num = manage_log_files()
log_filename = f"stock_data_server_{next_num}.log"

# File handler
file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Logging to file: {log_filename}")

class StockTradingBot:
    def __init__(self, data_server_url: str = "http://localhost:8000/ticker_data/api/ticker_data", 
                 trade_server_url: str = "http://localhost:5002"):
        self.data_server_url = data_server_url
        self.trade_server_url = trade_server_url
        self.running = False
        self.pivot_entry_time = None  # Track when price first entered pivot range
        
    def get_ticker_data(self, symbol: str) -> Optional[List[Dict]]:
        """Get recent intraday-like historical data for a ticker from Django ticker_data API.

        Uses historical_5m endpoint and normalizes rows to include currentPrice + timestamp.
        """
        try:
            response = requests.get(f"{self.data_server_url}/historical_5m/{symbol}")
            if response.status_code == 200:
                data = response.json()
                rows = data.get('data', [])

                normalized = []
                cumulative_by_day = defaultdict(int)
                for r in rows:
                    ts = r.get('timestamp')
                    if not ts:
                        continue

                    day_key = str(ts)[:10]
                    bar_volume = int(r.get('volume', 0) or 0)
                    cumulative_by_day[day_key] += bar_volume

                    normalized.append({
                        'timestamp': ts,
                        'currentPrice': r.get('close'),
                        'volume': cumulative_by_day[day_key],
                        'dayHigh': r.get('high'),
                        'dayLow': r.get('low'),
                    })

                return normalized
            else:
                logger.error(f"Failed to get data for {symbol}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get the latest data point for a ticker"""
        try:
            response = requests.get(f"{self.data_server_url}/data/{symbol}/latest")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get latest data for {symbol}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {str(e)}")
            return None

    def get_minutes_since_market_open(self) -> Optional[int]:
        """Get the number of minutes since market opened today"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Check if market is currently open
        if not is_market_open():
            return None
        
        # Get today's market open time
        market_open_today = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Calculate minutes since market opened
        minutes_since_open = (now - market_open_today).total_seconds() / 60
        return int(minutes_since_open)
    
    def calculate_volume_increase_in_timeframe(self, data: List[Dict], minutes: int) -> Optional[int]:
        """Calculate volume increase in the last X minutes, with robust gap/anomaly handling.

        Key robustness changes:
        - Removed unsafe fallback to the earliest record (often 0 at market open) when cutoff
          baseline was missing.
        - Require a data point at/before cutoff and at least one after cutoff; otherwise return None.
        - Detect suspicious 0 baselines far from session start (indicates truncated history) and abort.
        - Guard against intraday volume resets (negative deltas) by returning None.
        """
        logger.info(f"   Calculating volume increase for timeframe: {minutes} minutes")

        if minutes == -1:  # Entire day
            total_volume = sum(r.get('volume', 0) for r in data if r.get('volume') is not None)
            logger.info(f"   Total daily volume calculated: {total_volume}")
            return total_volume if total_volume > 0 else None

        minutes_since_open = self.get_minutes_since_market_open()
        if minutes_since_open is not None and minutes_since_open < minutes:
            logger.info(
                f"   Market has only been open for {minutes_since_open} minutes, adjusting timeframe from {minutes} to {minutes_since_open} minutes"
            )
            minutes = minutes_since_open

        now = datetime.now()
        cutoff_time = now - timedelta(minutes=minutes)

        # Build & sort timestamped data
        timestamped_data: List[Tuple[datetime, Dict]] = []
        for record in data:
            try:
                ts = record.get('timestamp')
                if not ts:
                    continue
                ts_work = ts
                if ts_work.endswith('Z'):
                    ts_work = ts_work[:-1]
                elif '+' in ts_work:
                    ts_work = ts_work.split('+')[0]
                elif ts_work.endswith('+00:00'):
                    ts_work = ts_work[:-6]
                try:
                    rec_time = datetime.fromisoformat(ts_work)
                except ValueError:
                    rec_time = datetime.strptime(ts_work, '%Y-%m-%dT%H:%M:%S.%f')
                timestamped_data.append((rec_time, record))
            except Exception as e:
                logger.debug(f"Failed parsing timestamp {record.get('timestamp','?')}: {e}")
                continue

        if not timestamped_data:
            logger.info("   No timestamped records available")
            return None

        timestamped_data.sort(key=lambda x: x[0])
        logger.info(
            f"   First record and last in timeframe in format {timestamped_data[0][0].strftime('%H:%M:%S')} | Price: {timestamped_data[0][1].get('currentPrice')} | Volume: {timestamped_data[0][1].get('volume')}"
        )
        logger.info(
            f"   Last record in timeframe in format {timestamped_data[-1][0].strftime('%H:%M:%S')} | Price: {timestamped_data[-1][1].get('currentPrice')} | Volume: {timestamped_data[-1][1].get('volume')}"
        )

        # Locate baseline (last record <= cutoff)
        left, right = 0, len(timestamped_data) - 1
        last_le_index = -1
        while left <= right:
            mid = (left + right) // 2
            if timestamped_data[mid][0] <= cutoff_time:
                last_le_index = mid
                left = mid + 1
            else:
                right = mid - 1

        if last_le_index == -1:
            earliest_time = timestamped_data[0][0]
            gap_minutes = (cutoff_time - earliest_time).total_seconds() / 60.0
            if gap_minutes > 5:
                logger.warning(
                    f"No data at/before cutoff ({cutoff_time.strftime('%H:%M:%S')}); earliest record {earliest_time.strftime('%H:%M:%S')} ({gap_minutes:.1f}m gap). Returning None."
                )
                return None
            logger.info(
                "Cutoff precedes earliest data but within 5m of start; using earliest volume as baseline."
            )
            volume_at_cutoff = (timestamped_data[0][1].get('volume') or 0)
        else:
            volume_at_cutoff = timestamped_data[last_le_index][1].get('volume')

        # Ensure at least one record strictly after cutoff
        has_after_cutoff = any(ts > cutoff_time for ts, _ in timestamped_data)
        if not has_after_cutoff:
            logger.warning(
                f"No records after cutoff ({cutoff_time.strftime('%H:%M:%S')}); cannot compute increase. Returning None."
            )
            return None

        # Latest non-null volume
        current_volume = None
        for ts, rec in reversed(timestamped_data):
            v = rec.get('volume')
            if v is not None:
                current_volume = v
                break

        if volume_at_cutoff is None or current_volume is None:
            logger.info(
                f"Insufficient data (volume_at_cutoff={volume_at_cutoff}, current_volume={current_volume}) for timeframe {minutes}m"
            )
            return None

        earliest_time = timestamped_data[0][0]
        minutes_since_earliest_cutoff = (cutoff_time - earliest_time).total_seconds() / 60.0
        if (
            minutes_since_earliest_cutoff > 30
            and volume_at_cutoff == 0
            and current_volume > 10000
        ):
            logger.warning(
                "Suspicious 0 baseline far from session start (history likely truncated); returning None."
            )
            return None

        logger.info(f"   Volume at cutoff ({cutoff_time.strftime('%H:%M:%S')}): {volume_at_cutoff}")
        logger.info(f"   Current/latest volume: {current_volume}")
        delta = current_volume - volume_at_cutoff
        if delta < 0:
            logger.warning(
                f"Volume decreased from {volume_at_cutoff} to {current_volume}; treating as anomaly and returning None."
            )
            return None
        logger.info(
            f"Volume increase in last {minutes} minutes: {delta} (from {volume_at_cutoff} to {current_volume})"
        )
        return delta
    
    def check_volume_requirements(self, data: List[Dict], volume_requirements: List[Tuple[int, int]], 
                             volume_multiplier: float = 1.0) -> bool:
        """Check if volume requirements are met.

        Updated semantics (2025-09-03): A trade passes the volume filter if ANY of the configured
        volume requirements passes (logical OR) instead of requiring ALL (previous AND logic).
        If no volume requirements are provided we pass by default.
        """
        if not volume_requirements:
            logger.info("   No volume requirements specified - PASSED")
            return True

        logger.info(f"   Checking {len(volume_requirements)} volume requirement(s) with {volume_multiplier}x multiplier (OR logic):")

        any_passed = False
        requirement_results = []  # Collect for summary logging
        for i, (minutes, required_volume) in enumerate(volume_requirements, 1):
            actual_volume_increase = self.calculate_volume_increase_in_timeframe(data, minutes)
            logger.info(f"   Raw calculated increase for {minutes if minutes != -1 else 'day'}: {actual_volume_increase}")

            if actual_volume_increase is None:
                logger.info(f"   Requirement {i}: Could not calculate volume increase for {minutes} minutes - FAILED")
                requirement_results.append(False)
                continue

            adjusted_required = int(required_volume * volume_multiplier)
            passed = actual_volume_increase >= adjusted_required
            requirement_results.append(passed)
            any_passed = any_passed or passed

            timeframe_str = "entire day" if minutes == -1 else f"{minutes} minutes"
            logger.info(f"   Requirement {i} ({timeframe_str}): {actual_volume_increase:,} >= {adjusted_required:,} - {'PASSED' if passed else 'FAILED'}")

        logger.info("   Result (OR across requirements): %s" % ("PASSED" if any_passed else "FAILED"))
        # Additional detail: how many passed
        passed_count = sum(1 for r in requirement_results if r)
        logger.info(f"   {passed_count}/{len(requirement_results)} requirement(s) passed under OR logic")
        return any_passed
    
    def check_price_breakout(self, data: List[Dict], current_price: float,
                              lookback_minutes: int = 60,
                              exclude_recent_minutes: float = 1.0) -> bool:
        """Return True if ANY price in the recent excluded window is strictly
        greater than ALL prices in the earlier (prior) window.

        Windows:
            PRIOR  WINDOW: (now - lookback_minutes, now - exclude_recent_minutes]
            RECENT WINDOW: (now - exclude_recent_minutes, now]

        We succeed if: max(recent) > max(prior).

        Args:
            data: Historical records (needs 'timestamp' & 'currentPrice').
            current_price: Kept for backward compatibility & logging; NOT the only
                           candidate anymore.
            lookback_minutes: Total minutes considered (must be > 0).
            exclude_recent_minutes: Size of the RECENT window boundary; if 0 the
                                    recent window collapses and breakout fails.
        """
        if lookback_minutes <= 0:
            logger.info("Breakout check skipped - non positive lookback_minutes")
            return False
        if exclude_recent_minutes < 0:
            logger.info("Breakout check skipped - exclude_recent_minutes negative")
            return False

        now = datetime.now()
        prior_start = now - timedelta(minutes=lookback_minutes)
        prior_end = now - timedelta(minutes=exclude_recent_minutes)
        recent_start = prior_end  # recent window is (recent_start, now]
        recent_end = now

        if prior_end <= prior_start:
            logger.info("Breakout check skipped - prior_end <= prior_start (invalid window config)")
            return False
        if recent_end <= recent_start:
            logger.info("Breakout check: exclude_recent_minutes too large (no recent window)")
            return False

        prior_prices = []
        recent_prices = []
        parse_failures = 0
        total_considered = 0
        for record in data:
            try:
                ts = record.get('timestamp')
                if not ts:
                    continue
                ts_work = ts
                if ts_work.endswith('Z'):
                    ts_work = ts_work[:-1]
                elif '+' in ts_work:
                    ts_work = ts_work.split('+')[0]
                elif ts_work.endswith('+00:00'):
                    ts_work = ts_work[:-6]
                try:
                    rec_time = datetime.fromisoformat(ts_work)
                except ValueError:
                    rec_time = datetime.strptime(ts_work, '%Y-%m-%dT%H:%M:%S.%f')
                price = record.get('currentPrice')
                if price is None:
                    continue
                # Classify into windows
                if prior_start < rec_time <= prior_end:
                    prior_prices.append(price)
                    total_considered += 1
                elif recent_start < rec_time <= recent_end:
                    recent_prices.append(price)
                    total_considered += 1
            except Exception:
                parse_failures += 1
                continue

        if not prior_prices:
            if recent_prices or current_price is not None:
                logger.info(
                    "Breakout check: PASS (insufficient prior window prices; "
                    f"auto-pass enabled) lookback={lookback_minutes} exclude_recent={exclude_recent_minutes} "
                    f"recent_pts={len(recent_prices)} current_price={current_price}"
                )
                return True
            else:
                logger.info(
                    "Breakout check: FAIL (no prior AND no usable recent prices) "
                    f"lookback={lookback_minutes} exclude_recent={exclude_recent_minutes}"
                )
                return False
        if not recent_prices:
            logger.info(f"Breakout check: FAIL (no recent window prices) lookback={lookback_minutes} exclude_recent={exclude_recent_minutes}")
            return False

        prior_high = max(prior_prices)
        recent_high = max(recent_prices)
        is_breakout = recent_high > prior_high
        logger.info(
            "Breakout check (ANY recent > ALL prior): "
            f"prior_high={prior_high:.4f} | recent_high={recent_high:.4f} | "
            f"prior_pts={len(prior_prices)} recent_pts={len(recent_prices)} | "
            f"parse_failures={parse_failures} considered={total_considered} | "
            f"prior_window=({prior_start.strftime('%H:%M:%S')} -> {prior_end.strftime('%H:%M:%S')}) | "
            f"recent_window=({recent_start.strftime('%H:%M:%S')} -> {recent_end.strftime('%H:%M:%S')}) | "
            f"current_price={current_price if current_price is not None else 'NA'} | RESULT={'PASSED' if is_breakout else 'FAILED'}"
        )
        return is_breakout
    
    def check_day_high_condition(self, current_price: float, day_high: float, max_percent_off: float = 0.5) -> bool:
        """Check if current price is at most max_percent_off% down from day's high"""
        if day_high is None or current_price is None:
            logger.info(f"   Day high condition check failed - missing data: current_price={current_price}, day_high={day_high}")
            return False
        
        max_drop = day_high * (max_percent_off / 100.0)
        min_acceptable_price = day_high - max_drop
        current_drop = day_high - current_price
        current_drop_percent = (current_drop / day_high) * 100
        
        condition_met = current_price >= min_acceptable_price
        
        logger.info(f"   Day high: {day_high:.4f}")
        logger.info(f"   Current price: {current_price:.4f}")
        logger.info(f"   Current drop: {current_drop:.4f} ({current_drop_percent:.2f}%)")
        logger.info(f"   Max allowed drop: {max_drop:.4f} ({max_percent_off}%)")
        logger.info(f"   Min acceptable price: {min_acceptable_price:.4f}")
        logger.info(f"   Condition: {'PASSED' if condition_met else 'FAILED'}")
        
        return condition_met
    
    def check_day_low_condition(self, day_low: float, max_day_low: float = None, min_day_low: float = None) -> bool:
        """Check if day's low is within acceptable range"""
        if day_low is None:
            logger.info(f"   Day low condition check skipped - missing day_low data")
            return True  # Skip check if no data
        
        conditions_passed = True
        
        # Check maximum day low
        if max_day_low is not None:
            max_condition_met = day_low <= max_day_low
            logger.info(f"   Day low: {day_low:.4f}")
            logger.info(f"   Max allowed day low: {max_day_low:.4f}")
            logger.info(f"   Max condition: {'PASSED' if max_condition_met else 'FAILED'}")
            conditions_passed = conditions_passed and max_condition_met
        
        # Check minimum day low
        if min_day_low is not None:
            min_condition_met = day_low >= min_day_low
            logger.info(f"   Day low: {day_low:.4f}")
            logger.info(f"   Min required day low: {min_day_low:.4f}")
            logger.info(f"   Min condition: {'PASSED' if min_condition_met else 'FAILED'}")
            conditions_passed = conditions_passed and min_condition_met
        
        if max_day_low is None and min_day_low is None:
            logger.info(f"   Day low condition check skipped - no limits set")
            return True
        
        logger.info(f"   Overall day low condition: {'PASSED' if conditions_passed else 'FAILED'}")
        return conditions_passed
    
    def get_pivot_position(self, current_price: float, lower_price: float, higher_price: float) -> str:
        """Determine which part of the pivot range the current price is in"""
        pivot_range = higher_price - lower_price
        price_position = (current_price - lower_price) / pivot_range
        
        if price_position <= 0.33:
            return "lower"
        elif price_position <= 0.66:
            return "middle"
        else:
            return "upper"
    
    def should_apply_time_in_pivot_requirement(self, pivot_position: str, time_in_pivot_positions: List[str]) -> bool:
        """Check if time-in-pivot requirement should be applied for current position"""
        if not time_in_pivot_positions:
            return False  # No requirement if no positions specified
        
        if "any" in time_in_pivot_positions:
            return True
        
        return pivot_position in time_in_pivot_positions
    
    def check_time_in_pivot_requirement(self, current_price: float, lower_price: float, higher_price: float,
                                       time_in_pivot_seconds: int, time_in_pivot_positions: List[str]) -> bool:
        """Check if price has been in specified pivot positions for required time"""
        if time_in_pivot_seconds <= 0:
            return True  # No time requirement
        
        current_time = datetime.now()
        
        # Check if price is currently in pivot range
        if current_price < lower_price or current_price > higher_price:
            self.pivot_entry_time = None
            logger.info(f"Price {current_price} not in pivot range, resetting timer")
            return False
        
        # Get current pivot position
        pivot_position = self.get_pivot_position(current_price, lower_price, higher_price)
        
        # Check if we need to apply time requirement for this position
        if not self.should_apply_time_in_pivot_requirement(pivot_position, time_in_pivot_positions):
            logger.info(f"Time-in-pivot requirement not applicable for position '{pivot_position}' - condition passed")
            return True
        
        # If we don't have an entry time, set it now
        if self.pivot_entry_time is None:
            self.pivot_entry_time = current_time
            logger.info(f"Price entered pivot range at position '{pivot_position}' at {current_time.strftime('%H:%M:%S')}")
        
        # Check if we've been in the pivot long enough
        time_in_pivot = (current_time - self.pivot_entry_time).total_seconds()
        condition_met = time_in_pivot >= time_in_pivot_seconds
        
        logger.info(f"Time-in-pivot check: position='{pivot_position}', time_in_pivot={time_in_pivot:.1f}s, "
                   f"required={time_in_pivot_seconds}s, met={condition_met}")
        
        return condition_met
    
    def execute_trade(self, ticker: str, lower_price: float, higher_price: float,
                      request_lower_price: Optional[float] = None,
                      request_higher_price: Optional[float] = None) -> bool:
        """Execute the trade by sending POST request to trade server"""
        try:
            # Use override prices if provided (logged for transparency)
            effective_lower = request_lower_price if request_lower_price is not None else lower_price
            effective_higher = request_higher_price if request_higher_price is not None else higher_price
            if request_lower_price is not None or request_higher_price is not None:
                logger.info(
                    f"Using overridden request trade prices: lower={effective_lower} (orig {lower_price}), higher={effective_higher} (orig {higher_price})"
                )
            payload = {"ticker": ticker, "lower_price": effective_lower, "higher_price": effective_higher}
            
            response = requests.post(f"{self.trade_server_url}/execute_trade", json=payload)
            
            if response.status_code == 200:
                logger.info(f"Trade executed successfully for {ticker}")
                return True
            else:
                logger.error(f"Trade execution failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            time.sleep(10)
            return False
    
    def monitor_and_trade(self, ticker: str, lower_price: float, higher_price: float,
                 volume_requirements: List[Tuple[int, int]], pivot_adjustment: float = 0.0,
                 day_high_max_percent_off: float = 0.5,
                 time_in_pivot_seconds: int = 0, time_in_pivot_positions: List[str] = None, 
                 volume_multipliers: List[float] = None, max_day_low: float = None, 
                 min_day_low: float = None,
                 wait_after_open_minutes: float = 0.0,
                 breakout_lookback_minutes: int = 60,
                 breakout_exclude_minutes: float = 1.0,
                 start_minutes_before_close: float = None,
                 stop_minutes_before_close: float = 0.0,
                 request_lower_price: Optional[float] = None,
                 request_higher_price: Optional[float] = None):
        """Main monitoring and trading logic"""
        adjusted_higher_price = higher_price * (1 + pivot_adjustment)
        
        if time_in_pivot_positions is None:
            time_in_pivot_positions = []
        
        logger.info(f"Starting monitoring for {ticker}")
        logger.info(f"Pivot range: {lower_price} - {adjusted_higher_price}")
        logger.info(f"Volume requirements: {volume_requirements}")
        logger.info(f"Volume multipliers: {volume_multipliers}")
        logger.info(f"Pivot adjustment: {pivot_adjustment*100}%")
        logger.info(f"Breakout settings: lookback={breakout_lookback_minutes}m, exclude_recent={breakout_exclude_minutes}m")
        logger.info(f"Day high max percent off: {day_high_max_percent_off}%")
        logger.info(f"Time-in-pivot requirement: {time_in_pivot_seconds}s for positions {time_in_pivot_positions}")
        logger.info(f"Max day low: {max_day_low}")
        logger.info(f"Min day low: {min_day_low}")
        logger.info(f"Wait after open minutes: {wait_after_open_minutes}")
        logger.info(f"Late-day start (minutes before close): {start_minutes_before_close}")
        logger.info(f"Late-day stop (minutes before close): {stop_minutes_before_close}")
        logger.info(f"Override request trade prices: lower={request_lower_price} higher={request_higher_price}")
        if start_minutes_before_close is not None and stop_minutes_before_close is not None:
            if start_minutes_before_close <= stop_minutes_before_close:
                logger.warning("Configuration may result in zero/negative trading window: start_minutes_before_close <= stop_minutes_before_close")
        logger.info("Legacy average momentum removed; using breakout only")

        wait_for_market_open()

        # Optional initial wait period after market open (float minutes allowed)
        if wait_after_open_minutes and wait_after_open_minutes > 0:
            try:
                et = pytz.timezone('US/Eastern')
                while True:
                    now_et = datetime.now(et)
                    market_open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                    minutes_since_open_float = (now_et - market_open_dt).total_seconds() / 60.0
                    if minutes_since_open_float >= wait_after_open_minutes:
                        logger.info(f"Wait-after-open period ({wait_after_open_minutes} min) satisfied; starting monitoring loop.")
                        break
                    remaining = wait_after_open_minutes - minutes_since_open_float
                    logger.info(f"Waiting additional {remaining:.2f} minute(s) after open before starting monitoring.")
                    # Sleep up to 5 seconds or remaining time in seconds, whichever smaller but at least 1s
                    sleep_sec = max(1.0, min(5.0, remaining * 60.0))
                    time.sleep(sleep_sec)
            except Exception as e:
                logger.warning(f"Could not perform wait-after-open logic: {e}")

        self.running = True
            
        cycle_count = 0
        start_time = datetime.now()
        
        while self.running:
            try:
                cycle_count += 1
                cycle_start = datetime.now()
                logger.info(f"\n{'='*60}")
                logger.info(f"MONITORING CYCLE #{cycle_count} - {cycle_start.strftime('%H:%M:%S.%f')[:-3]}")
                logger.info(f"{'='*60}")
                
                # Check if market is still open
                if not is_market_open():
                    logger.info("Market has closed. Returning to wait for next market open...")
                    self.pivot_entry_time = None  # Reset pivot timer
                    wait_for_market_open()
                    continue

                # --- NEW: Late-day window enforcement ---
                try:
                    et = pytz.timezone('US/Eastern')
                    now_et = datetime.now(et)
                    market_close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                    minutes_to_close = (market_close_dt - now_et).total_seconds() / 60.0
                    # If start_minutes_before_close is set, only proceed when minutes_to_close <= start threshold
                    if start_minutes_before_close is not None:
                        if minutes_to_close > start_minutes_before_close:
                            h = int(minutes_to_close // 60)
                            m = int(minutes_to_close % 60)
                            h_target = int(start_minutes_before_close // 60)
                            m_target = int(start_minutes_before_close % 60)
                            window_str = (f"{h}h {m}m" if h else f"{m}m")
                            target_str = (f"{h_target}h {m_target}m" if h_target else f"{m_target}m")
                            logger.info(
                                f"Late-day start restriction active: {window_str} until close (> {target_str}). Waiting..."
                            )
                            time.sleep(5)
                            continue
                    # If we are inside the stop window, cease trading attempts for the day
                    if stop_minutes_before_close is not None and stop_minutes_before_close > 0:
                        if minutes_to_close <= stop_minutes_before_close:
                            h = int(minutes_to_close // 60)
                            m = int(minutes_to_close % 60)
                            h_stop = int(stop_minutes_before_close // 60)
                            m_stop = int(stop_minutes_before_close % 60)
                            window_str = (f"{h}h {m}m" if h else f"{m}m")
                            stop_str = (f"{h_stop}h {m_stop}m" if h_stop else f"{m_stop}m")
                            logger.info(
                                f"Stop window reached: {window_str} until close (<= {stop_str}). Waiting for next session..."
                            )
                            # Wait until next market open
                            self.pivot_entry_time = None
                            wait_for_market_open()
                            continue
                except Exception as e:
                    logger.warning(f"Late-day window logic error (continuing anyway): {e}")

                # (Removed historical average momentum warm-up logic)

                # (Removed min_time_since_open_minutes logic; superseded by wait_after_open_minutes initial delay)
                
                # Get current data
                latest_data = self.get_latest_data(ticker)
                if not latest_data:
                    logger.warning(f"No latest data available for {ticker}")
                    time.sleep(5)
                    continue
                
                current_price = latest_data.get('currentPrice')
                day_high = latest_data.get('dayHigh')
                day_low = latest_data.get('dayLow')
                
                if current_price is None:
                    logger.warning(f"No current price available for {ticker}")
                    time.sleep(5)
                    continue
                
                if current_price < lower_price or current_price > adjusted_higher_price:
                    if current_price < lower_price:
                        logger.info(f"Price {current_price} is BELOW pivot range (min: {lower_price}, max: {adjusted_higher_price}) - difference: {lower_price - current_price:.4f}")
                    else:
                        logger.info(f"Price {current_price} is ABOVE pivot range (min: {lower_price}, max: {adjusted_higher_price}) - difference: {current_price - adjusted_higher_price:.4f}")
                    self.pivot_entry_time = None  # Reset timer when out of range
                    time.sleep(2)
                    continue

                logger.info(f"✓ Price {current_price} is IN pivot range [{lower_price}, {adjusted_higher_price}]")
                
                # Get historical data for analysis
                historical_data = self.get_ticker_data(ticker)
                if not historical_data:
                    logger.warning(f"No historical data available for {ticker}")
                    time.sleep(5)
                    continue
                
                ## Determine pivot position and volume multiplier
                pivot_position = self.get_pivot_position(current_price, lower_price, adjusted_higher_price)
                pivot_range = adjusted_higher_price - lower_price
                price_position_percent = ((current_price - lower_price) / pivot_range) * 100

                if volume_multipliers is None:
                    volume_multipliers = [1.0, 1.0, 1.0]

                if pivot_position == "lower":
                    volume_multiplier = volume_multipliers[0]
                elif pivot_position == "middle":
                    volume_multiplier = volume_multipliers[1]
                else:
                    volume_multiplier = volume_multipliers[2]


                logger.info(f"📊 PIVOT ANALYSIS:")
                logger.info(f"   Current price: {current_price}")
                logger.info(f"   Pivot range: {lower_price} - {adjusted_higher_price} (span: {pivot_range:.4f})")
                logger.info(f"   Position in range: {price_position_percent:.1f}% ({pivot_position} section)")
                logger.info(f"   Volume multiplier: {volume_multiplier}x")
                
                # Check all conditions
                conditions_met = True
                failed_conditions = []
                logger.info("=== CHECKING ALL CONDITIONS ===")

                # 1. Check day high condition
                logger.info("1. Checking day high condition...")
                if not self.check_day_high_condition(current_price, day_high, day_high_max_percent_off):
                    conditions_met = False
                    failed_conditions.append("day_high")
                    logger.info("   ❌ Day high condition FAILED")
                else:
                    logger.info("   ✓ Day high condition PASSED")
                    
                # 2. Check day low condition  
                if conditions_met:
                    logger.info("2. Checking day low condition...")
                    if not self.check_day_low_condition(day_low, max_day_low, min_day_low):  # Add min_day_low
                        conditions_met = False
                        failed_conditions.append("day_low")
                        logger.info("   ❌ Day low condition FAILED")
                    else:
                        logger.info("   ✓ Day low condition PASSED")
                else:
                    logger.info("2. Skipping day low check (previous condition failed)")


                # 3. Breakout condition
                if conditions_met:
                    logger.info("3. Checking breakout (price > prior interval high)...")
                    if not self.check_price_breakout(historical_data, current_price,
                                                      lookback_minutes=breakout_lookback_minutes,
                                                      exclude_recent_minutes=breakout_exclude_minutes):
                        conditions_met = False
                        failed_conditions.append("breakout")
                        logger.info("   ❌ Breakout FAILED")
                    else:
                        logger.info("   ✓ Breakout PASSED")
                else:
                    logger.info("3. Skipping breakout check (previous condition failed)")

                # 4. Check volume requirements
                if conditions_met:
                    logger.info("4. Checking volume requirements...")
                    if not self.check_volume_requirements(historical_data, volume_requirements, volume_multiplier):
                        conditions_met = False
                        failed_conditions.append("volume")
                        logger.info("   ❌ Volume requirements FAILED")
                    else:
                        logger.info("   ✓ Volume requirements PASSED")
                else:
                    logger.info("4. Skipping volume check (previous condition failed)")

                # 5. Check time-in-pivot requirement
                if conditions_met:
                    logger.info("5. Checking time-in-pivot requirement...")
                    if not self.check_time_in_pivot_requirement(current_price, lower_price, adjusted_higher_price,
                                                            time_in_pivot_seconds, time_in_pivot_positions):
                        conditions_met = False
                        failed_conditions.append("time_in_pivot")
                        logger.info("   ❌ Time-in-pivot requirement FAILED")
                    else:
                        logger.info("   ✓ Time-in-pivot requirement PASSED")
                else:
                    logger.info("5. Skipping time-in-pivot check (previous condition failed)")

                # Summary of results
                if conditions_met:
                    logger.info("🎉 ALL CONDITIONS MET! Executing trade...")
                    if self.execute_trade(ticker, lower_price, higher_price,
                                           request_lower_price=request_lower_price,
                                           request_higher_price=request_higher_price):
                        logger.info(f"✅ Trade executed successfully for {ticker}")
                        break
                    else:
                        logger.error(f"❌ Trade execution failed for {ticker}")
                else:
                    logger.info(f"❌ CONDITIONS NOT MET - Failed: {', '.join(failed_conditions)}")
            except KeyboardInterrupt:
                logger.info("Stopping due to keyboard interrupt")
                time.sleep(10)
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                time.sleep(5)
            
            # Throttle extremely fast cycles: if this monitoring cycle finished in under 1s
            # (i.e., little/no waiting occurred inside), sleep for 4 seconds to avoid
            # hammering the data/trade servers.
            try:
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                if cycle_duration < 1.0:
                    logger.info(f"Cycle duration {cycle_duration:.3f}s < 1s; sleeping 4s to throttle requests.")
                    time.sleep(4)
            except Exception as throttle_err:
                logger.debug(f"Throttle timing computation failed (continuing): {throttle_err}")
        
        self.running = False

def parse_volume_requirements(volume_args: List[str]) -> List[Tuple[int, int]]:
    """Parse volume requirement arguments"""
    requirements = []
    
    for arg in volume_args:
        try:
            if '=' in arg:
                time_part, volume_part = arg.split('=', 1)
                if time_part.lower() == 'day':
                    minutes = -1
                else:
                    minutes = int(time_part)
                volume = int(volume_part)
                requirements.append((minutes, volume))
            else:
                logger.error(f"Invalid volume requirement format: {arg}")
        except ValueError:
            logger.error(f"Invalid volume requirement format: {arg}")
            time.sleep(10)
    
    return requirements

def parse_pivot_positions(positions_str: str) -> List[str]:
    """Parse pivot position string into list of positions"""
    if not positions_str:
        return []
    
    valid_positions = ["lower", "middle", "upper", "any"]
    positions = [pos.strip().lower() for pos in positions_str.split(',')]
    
    # Validate positions
    for pos in positions:
        if pos not in valid_positions:
            logger.error(f"Invalid pivot position: {pos}. Valid options: {', '.join(valid_positions)}")
            return []
    
    return positions
  
def is_market_open() -> bool:
    """Check if the market is currently open (9:30 AM - 4:00 PM ET, Monday-Friday)"""
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    current_time = now.time()
    
    return market_open <= current_time <= market_close
  
def minutes_until_market_open() -> int:
    """Calculate minutes until next market open"""
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    
    # If it's weekend, calculate time until Monday 9:30 AM
    if now.weekday() > 4:  # Saturday or Sunday
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:  # Sunday
            days_until_monday = 1
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
    else:
        # It's a weekday
        market_open_today = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.time() < dt_time(9, 30):
            # Before market open today
            next_open = market_open_today
        else:
            # After market close today, next open is tomorrow (if it's a weekday)
            if now.weekday() == 4:  # Friday
                next_open = market_open_today + timedelta(days=3)  # Monday
            else:
                next_open = market_open_today + timedelta(days=1)  # Next day
    
    return int((next_open - now).total_seconds() / 60)

def wait_for_market_open():
    """Wait until market opens with precise timing"""
    while not is_market_open():
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Calculate next market open time
        if now.weekday() > 4:  # Weekend
            days_until_monday = 7 - now.weekday()
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
        else:
            # Weekday
            market_open_today = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now.time() < dt_time(9, 30):
                next_open = market_open_today
            else:
                # After market close, next open is next business day
                if now.weekday() == 4:  # Friday
                    next_open = market_open_today + timedelta(days=3)  # Monday
                else:
                    next_open = market_open_today + timedelta(days=1)
        
        # Calculate wait time
        wait_seconds = (next_open - now).total_seconds()
        
        if wait_seconds <= 0:
            break  # Market should be open, exit loop
            
        hours = int(wait_seconds // 3600)
        minutes = int((wait_seconds % 3600) // 60)
        
        sleep_time = min(get_adaptive_sleep_time(wait_seconds), wait_seconds)
        sleep_minutes = int(sleep_time // 60)
        sleep_seconds_remainder = int(sleep_time % 60)
        logger.info(f"Market closed. Waiting {hours}h {minutes}m until market open at {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')} (sleeping for {sleep_minutes}m {sleep_seconds_remainder}s)")
        
        # Sleep with adaptive timing based on distance from market open
        sleep_time = min(get_adaptive_sleep_time(wait_seconds), wait_seconds)
        time.sleep(sleep_time)
    
    logger.info("Market is now open!")
    
def get_adaptive_sleep_time(wait_seconds: float) -> float:
    """Calculate adaptive sleep time based on how long until market open"""
    if wait_seconds > 24 * 3600:  # More than 24 hours
        return 7200  # Sleep for 2 hours
    elif wait_seconds > 12 * 3600:  # More than 12 hours
        return 3600  # Sleep for 1 hour
    elif wait_seconds > 6 * 3600:  # More than 6 hours
        return 1800  # Sleep for 30 minutes
    elif wait_seconds > 3 * 3600:  # More than 3 hours
        return 600   # Sleep for 10 minutes
    elif wait_seconds > 1 * 3600:  # More than 1 hour
        return 240   # Sleep for 4 minutes
    else:
        return 60   # Sleep for 1 minute

def debug_timezone_info():
    import pytz
    from datetime import datetime
    
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    now_local = datetime.now()
    
    logger.info(f"DEBUG: Local system time: {now_local}")
    logger.info(f"DEBUG: Eastern time: {now_et}")
    logger.info(f"DEBUG: Market should be open: {9.5 <= now_et.hour + now_et.minute/60 <= 16}")




def main():
    parser = argparse.ArgumentParser(description='Stock Trading Bot')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('lower_price', type=float, help='Lower pivot price')
    parser.add_argument('higher_price', type=float, help='Higher pivot price')
    parser.add_argument('--volume', action='append', default=[], 
                       help='Volume requirements in format "minutes=volume" or "day=volume". Can be specified multiple times.')
    parser.add_argument('--pivot-adjustment', choices=['0.0', '0.5', '1.0'], default='0.0',
                       help='Increase upper pivot price by 0.0%, 0.5%, or 1.0%')
    # Removed legacy average momentum arguments
    parser.add_argument('--day-high-max-percent-off', type=float, default=0.5,
                       help='Maximum percentage the current price can be below day high (default: 0.5)')
    parser.add_argument('--time-in-pivot', type=int, default=0,
                       help='Required time in seconds that price must be in pivot range (default: 0 = no requirement)')
    parser.add_argument('--time-in-pivot-positions', type=str, default='',
                       help='Comma-separated list of pivot positions where time requirement applies. Options: lower, middle, upper, any (default: empty = no requirement)')
    parser.add_argument('--data-server', default='http://localhost:8000/ticker_data/api/ticker_data',
                       help='Django ticker_data API base URL')
    parser.add_argument('--trade-server', default='http://localhost:5002',
                       help='Trade server URL')
    parser.add_argument('--volume-multipliers', nargs=3, type=float, metavar=('LOWER', 'MIDDLE', 'UPPER'),
                    default=[1.0, 1.0, 1.0],
                    help='Volume multipliers for lower, middle, and upper pivot positions (default: 1.0 1.0 1.0)')
    parser.add_argument('--max-day-low', type=float, default=None,
                   help='Maximum day low price allowed (default: None = no limit)')
    parser.add_argument('--min-day-low', type=float, default=None,
                   help='Minimum day low price allowed (default: None = no limit)')
    parser.add_argument('--wait-after-open', type=float, default=0.0,
                   help='Additional float minutes to wait after market open before starting monitoring (default: 0 = no extra wait)')
    # Only breakout momentum retained
    parser.add_argument('--breakout-lookback-minutes', type=int, default=60,
                    help='Lookback window in minutes for breakout momentum (default: 60)')
    parser.add_argument('--breakout-exclude-minutes', type=float, default=1.0,
                    help='Minutes immediately before now to exclude when computing prior high (default: 1.0)')
    # --- NEW (2025-09-07): Optional request trade pivot override ---
    parser.add_argument('--request-lower-price', type=float, default=None,
                        help='If set, this lower price will be sent to the trade server instead of monitoring lower_price.')
    parser.add_argument('--request-higher-price', type=float, default=None,
                        help='If set, this higher price will be sent to the trade server instead of monitoring higher_price.')
    # --- NEW (2025-09-05): Late-day trading window controls ---
    parser.add_argument('--start-minutes-before-close', type=float, default=None,
                        help='Only allow initiating trades once time until market close is <= this many minutes (e.g. 60 = last hour). Default: None (no late-day start restriction).')
    parser.add_argument('--stop-minutes-before-close', type=float, default=0.0,
                        help='Stop initiating trades this many minutes before market close (e.g. 5 = no trades in final 5 minutes). Default: 0 (trade until close).')
    # momentum_required_at_open removed

    args = parser.parse_args()
    
    debug_timezone_info()

    # Parse volume requirements
    volume_requirements = parse_volume_requirements(args.volume)
    
    # Parse pivot positions
    time_in_pivot_positions = parse_pivot_positions(args.time_in_pivot_positions)
    
    # Convert pivot adjustment to decimal
    pivot_adjustment = float(args.pivot_adjustment) / 100.0
    
    # Create and run the bot
    bot = StockTradingBot(args.data_server, args.trade_server)
    
    try:
        # Attach override values (will be forwarded via dynamic attribute access below)
        bot.monitor_and_trade(
            ticker=args.ticker.upper(),
            lower_price=args.lower_price,
            higher_price=args.higher_price,
            volume_requirements=volume_requirements,
            pivot_adjustment=pivot_adjustment,
            day_high_max_percent_off=args.day_high_max_percent_off,
            time_in_pivot_seconds=args.time_in_pivot,
            time_in_pivot_positions=time_in_pivot_positions,
            volume_multipliers=args.volume_multipliers,
            max_day_low=args.max_day_low,
            min_day_low=args.min_day_low,
            wait_after_open_minutes=args.wait_after_open,
            breakout_lookback_minutes=args.breakout_lookback_minutes,
            breakout_exclude_minutes=args.breakout_exclude_minutes,
            start_minutes_before_close=args.start_minutes_before_close,
            stop_minutes_before_close=args.stop_minutes_before_close,
            request_lower_price=args.request_lower_price,
            request_higher_price=args.request_higher_price,
        )

        
        # Wait for user to stop the bot
        while bot.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot failed with error: {str(e)}")
        time.sleep(10)

if __name__ == "__main__":
    main()