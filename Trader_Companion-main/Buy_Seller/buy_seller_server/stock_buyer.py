import json
import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from flask import Flask, request, jsonify
import math
from queue import Queue
import uuid
import flask_cors
import requests
import urllib3
import os
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Database module for trade persistence
from db import init_db, add_trade as db_add_trade, find_trade_by_criteria as db_find_trade, \
    find_trade_by_id as db_find_trade_by_id, delete_trade as db_delete_trade, \
    trade_exists as db_trade_exists, get_all_trades as db_get_all_trades, get_trade_count, \
    save_risk_amount, load_risk_amount


@dataclass
class SellStopOrder:
    """Represents either a fixed-price stop or a percentage-below-fill stop.

    Exactly one of price or percent_below_fill should be provided.
    price: Absolute stop price.
    percent_below_fill: Percentage (e.g. 4 for 4%) below the eventual average fill price to place the stop.
    shares: Number of shares allocated to this stop.
    """
    price: Optional[float] = None
    shares: float = 0.0
    percent_below_fill: Optional[float] = None

@dataclass
class Trade:
    ticker: str
    shares: float
    risk_amount: float
    lower_price_range: float
    higher_price_range: float
    sell_stops: List[SellStopOrder]
    order_type: str = "MKT"
    adaptive_priority: Optional[str] = None
    timeout_seconds: int = 5
    trade_id: str = None
    
    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = str(uuid.uuid4())

class IBWebAPI:
    def __init__(self, base_url: Optional[str] = None):
        # Default assumes ibeam container mapping: host 5050 -> container 5000
        # Override with env var if you run it elsewhere.
        self.base_url = (base_url or os.getenv('IBKR_WEB_API_BASE_URL') or "https://localhost:5050/v1/api").rstrip('/')
        self.session = requests.Session()
        self.session.verify = False
        self.session.timeout = 30

    def get_contract_details(self, conid):
        """Get contract details for a given conid"""
        try:
            url = f"{self.base_url}/iserver/secdef/info?conid={conid}"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            print(f"❌ Failed to get contract details for conid {conid}: {response.status_code}, {response.text}")
            return None
        except Exception as e:
            print(f"❌ Contract details error: {str(e)}")
            return None

    def get_contract_algos(self, conid, add_description: bool = False, add_params: bool = False, algos: Optional[str] = None):
        """Get available algos for a contract ID."""
        try:
            url = f"{self.base_url}/iserver/contract/{conid}/algos"
            params = {}
            if add_description:
                params['addDescription'] = 1
            if add_params:
                params['addParams'] = 1
            if algos:
                params['algos'] = algos

            response = self.session.get(url, params=params, timeout=30)
            if response.status_code != 200:
                print(f"❌ Failed to get algos for conid {conid}: {response.status_code}, {response.text}")
                return None

            payload = response.json()
            if isinstance(payload, dict):
                return payload.get('algos', [])
            if isinstance(payload, list):
                return payload
            return []
        except Exception as e:
            print(f"❌ Contract algos error for conid {conid}: {str(e)}")
            return None
        
    def is_connected(self):
        try:
            response = self.session.get(f"{self.base_url}/iserver/auth/status", timeout=10)
            return response.status_code == 200 and response.json().get('authenticated', False)
        except:
            return False
        
    def get_accounts(self):
        """Get account information"""
        try:
            response = self.session.get(f"{self.base_url}/iserver/accounts", timeout=10)
            print(f"   📊 Accounts response: {response.status_code}")
            if response.status_code == 200:
                print(f"   📊 Accounts: {response.json()}")
            else:
                print(f"   📊 Accounts error: {response.text}")
            return response
        except Exception as e:
            print(f"   ❌ Get accounts error: {str(e)}")
            return None
    
    def place_order(self, conid, order_data):
        """Place order via Web API"""
        # First get the account ID
        accounts_response = self.session.get(f"{self.base_url}/iserver/accounts", timeout=10)
        if accounts_response.status_code != 200:
            print(f"   ❌ Failed to get accounts: {accounts_response.status_code}")
            return accounts_response
        
        accounts_data = accounts_response.json()
        account_id = accounts_data.get('selectedAccount')
        if not account_id:
            print(f"   ❌ No selected account found")
            return accounts_response
        
        url = f"{self.base_url}/iserver/account/{account_id}/orders"
        payload = {
            "orders": [{
                "acctId": account_id,
                "conid": int(conid),  # Ensure it's an integer
                "secType": f"{int(conid)}:STK",
                "orderType": order_data["orderType"],
                "side": order_data["side"],
                "quantity": order_data["quantity"],
                "tif": order_data.get("tif", "DAY")  # Default to DAY if not provided
            }]
        }
        
        # Only add price fields if they exist
        if order_data.get("price") is not None:
            payload["orders"][0]["price"] = order_data["price"]
        if order_data.get("auxPrice") is not None:
            payload["orders"][0]["auxPrice"] = order_data["auxPrice"]
        if order_data.get("algoStrategy"):
            payload["orders"][0]["algoStrategy"] = order_data["algoStrategy"]
        if order_data.get("algoParams"):
            payload["orders"][0]["algoParams"] = order_data["algoParams"]

        # Compatibility fields for API variants that expect strategy/strategyParameters.
        if order_data.get("strategy"):
            payload["orders"][0]["strategy"] = order_data["strategy"]
        if order_data.get("strategyParameters"):
            payload["orders"][0]["strategyParameters"] = order_data["strategyParameters"]
        
        print(f"   📤 Sending order to: {url}")
        print(f"   📋 Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = self.session.post(url, json=payload, timeout=30)
            print(f"   📥 Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"   📥 Response headers: {dict(response.headers)}")
                print(f"   📥 Response text: {response.text}")
            return response
        except Exception as e:
            print(f"   ❌ Request exception: {str(e)}")
            raise
    
    def get_contract_id(self, symbol):
        """Get contract ID for a symbol with smart filtering"""
        url = f"{self.base_url}/iserver/secdef/search"
        payload = {"symbol": symbol, "secType": "STK"}
        try:
            response = self.session.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Filter and prioritize
                    # 1. Exact symbol match
                    # 2. Currency = USD
                    # 3. US Exchanges
                    
                    candidates = []
                    print(f"   🔎 Search returned {len(data)} results. Inspecting first 5:")
                    for i, item in enumerate(data[:5]):
                        # Safely get fields with defaults
                        symbol_val = item.get('symbol')
                        desc_val = item.get('description')
                        sec_type_val = item.get('secType')
                        currency_val = item.get('currency')
                        conid_val = item.get('conid')
                        
                        print(f"      Result {i+1}: {symbol_val} | {desc_val} | {sec_type_val} | {currency_val} | ID: {conid_val}")
                        
                        # Relaxed validation - check both secType and type
                        sec_type = sec_type_val or item.get('type')
                        if sec_type and sec_type != 'STK' and sec_type != 'Stock':
                            continue
                            
                        score = 0
                        # Exact symbol match
                        if symbol_val == symbol:
                            score += 10
                        
                        # Currency match (assume USD preference)
                        if currency_val == 'USD':
                            score += 5
                        
                        # Exchange preference
                        # Handle None description safely
                        desc = (desc_val or '').upper()
                        if 'NASDAQ' in desc or 'NYSE' in desc or 'AMEX' in desc:
                            score += 3
                        elif 'ARCA' in desc or 'BATS' in desc:
                            score += 2
                            
                        candidates.append((score, item))
                    
                    # Sort by score descending
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    if candidates:
                        best_match = candidates[0][1]
                        # Safely get description for logging
                        best_desc = (best_match.get('description') or 'Unknown').upper()
                        print(f"   🔍 Selected contract for {symbol}: {best_desc} (ID: {best_match.get('conid')})")
                        return best_match.get("conid")
                    
                    # Fallback to first item if no candidates passed filter
                    print(f"   ⚠️ No ideal match found for {symbol}, using first result")
                    return data[0].get("conid")
                    
            return None
        except Exception as e:
            print(f"   ❌ Contract search error: {str(e)}")
            return None
    
    def get_order_status(self, order_id=None):
        """Get full order status list.

        Some accounts return "Allocation Id missing" on per-order endpoints,
        so we consistently use the full orders endpoint and filter client-side.
        """
        url = f"{self.base_url}/iserver/account/orders"
        return self.session.get(url, timeout=30)

    def find_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Return order dict for a given order_id from full order list."""
        try:
            response = self.get_order_status()
            if response.status_code != 200:
                return None

            payload = response.json()
            orders = []
            if isinstance(payload, dict):
                if isinstance(payload.get('orders'), list):
                    orders = payload.get('orders', [])
                elif isinstance(payload.get('data'), list):
                    orders = payload.get('data', [])
            elif isinstance(payload, list):
                orders = payload

            for order in orders:
                oid = order.get('orderId') or order.get('order_id') or order.get('id')
                if str(oid) == str(order_id):
                    return order
            return None
        except Exception:
            return None
    
    def cancel_order(self, order_id):
        """Cancel an order with proper account context"""
        try:
            payload = {}
            # Optionally include allocationId if present in order details
            order_data = self.find_order_by_id(order_id)
            if isinstance(order_data, dict):
                try:
                    allocation_id = order_data.get("allocationId")
                    if allocation_id:
                        payload['allocationId'] = allocation_id
                        print(f"📋 Using allocationId from order: {allocation_id}")
                except Exception:
                    pass

            
            # Get account information to retrieve allocationId or account context
            accounts_response = self.session.get(f"{self.base_url}/iserver/accounts", timeout=10)
            if accounts_response.status_code != 200:
                print(f"❌ Failed to get accounts for cancellation: {accounts_response.status_code}, {accounts_response.text}")
                return accounts_response

            accounts_data = accounts_response.json()
            account_id = accounts_data.get('selectedAccount')
            if not account_id:
                print(f"❌ No selected account found for cancellation")
                return accounts_response

            # Construct the cancellation URL
            url = f"{self.base_url}/iserver/account/{account_id}/order/{order_id}"

            
            # Optionally include allocationId if required
            if 'allocationProfile' in accounts_data:
                allocation_id = accounts_data.get('allocationProfile', {}).get('id')
                if allocation_id:
                    payload['allocationId'] = allocation_id
                    print(f"📋 Using allocationId: {allocation_id}")

            print(f"📤 Sending cancel request to: {url}")
            print(f"📋 Cancel payload: {json.dumps(payload, indent=2)}")
            
            response = self.session.delete(url, json=payload, timeout=30)
            print(f"📥 Cancel response status: {response.status_code}, Body: {response.text}")
            
            if response.status_code in [200, 201]:
                return response
            else:
                print(f"❌ Cancel failed: {response.status_code}, {response.text}")
                return response

        except Exception as e:
            print(f"❌ Cancel order error: {str(e)}")
            return {'status_code': 500, 'text': str(e)}

class StockTradingServer:
    def __init__(self):
        # Initialize database
        init_db()
        
        # Load persisted risk amount from database (or 0.0 if not set)
        self.available_risk: float = load_risk_amount()
        self.error_log: List[Dict] = []
        
        self.server_start_time = time.time()
        self.last_trade_time = None
        
        # Threading for sequential processing
        self.request_queue = Queue()
        self.processing_thread = None
        self.is_processing = False
        self.server_running = True
        
        # IBKR Web API connection
        self.ib_api = None
        
        # Note: Duplicate trade prevention is now handled by database:
        # A trade is DELETED from DB before order is sent, so it cannot be executed twice
        
        # Start processing thread
        self.start_processing_thread()
        
    def _remove_trade_internal(self, data: dict) -> dict:
        """Internal method to remove a trade from database"""
        try:
            trade_id = data.get('trade_id')
            ticker = data.get('ticker')
            lower_price = data.get('lower_price')
            higher_price = data.get('higher_price')
            
            # Find trade by ID first, then by criteria as fallback
            trade_data = None
            
            if trade_id:
                trade_data = db_find_trade_by_id(trade_id)
            
            # If not found by ID, try to find by criteria
            if trade_data is None and ticker and lower_price is not None and higher_price is not None:
                trade_data = db_find_trade(ticker, lower_price, higher_price)
            
            if trade_data is None:
                return {'success': False, 'error': 'Trade not found'}
            
            # Remove the trade from database
            if not db_delete_trade(trade_data['trade_id']):
                return {'success': False, 'error': 'Failed to delete trade from database'}
            
            return {
                'success': True,
                'message': f"Trade removed: {trade_data['ticker']} (${trade_data['lower_price_range']} - ${trade_data['higher_price_range']})",
                'removed_trade': {
                    'trade_id': trade_data['trade_id'],
                    'ticker': trade_data['ticker'],
                    'shares': trade_data['shares'],
                    'risk_amount': trade_data['risk_amount'],
                    'lower_price_range': trade_data['lower_price_range'],
                    'higher_price_range': trade_data['higher_price_range']
                },
                'available_risk': self.available_risk
            }
            
        except Exception as e:
            error_msg = f"Error removing trade: {str(e)}"
            self._log_error("REMOVE_TRADE_ERROR", data.get('ticker', 'unknown'), error_msg, data)
            return {'success': False, 'error': error_msg}


    def remove_trade(self, trade_id: str = None, ticker: str = None, lower_price: float = None, higher_price: float = None) -> dict:
        """Remove a trade by ID or criteria"""
        return self._queue_request('remove_trade', {
            'trade_id': trade_id,
            'ticker': ticker,
            'lower_price': lower_price,
            'higher_price': higher_price
        })

    
    def start_processing_thread(self):
        """Start the sequential processing thread"""
        self.processing_thread = threading.Thread(target=self._process_requests)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_requests(self):
        """Process requests sequentially"""
        while self.server_running:
            try:
                # Get next request (blocks until available)
                request_data = self.request_queue.get(timeout=1)
                if request_data is None:
                    continue
                
                self.is_processing = True
                
                # Process the request
                request_type = request_data['type']
                response_queue = request_data['response_queue']
                
                try:
                    if request_type == 'execute_trade':
                        result = self._execute_trade_internal(request_data['data'])
                    elif request_type == 'add_trade':
                        result = self._add_trade_internal(request_data['data'])
                    elif request_type == 'get_status':
                        result = self._get_status_internal()
                    elif request_type == 'remove_trade':
                        result = self._remove_trade_internal(request_data['data'])
                    else:
                        result = {'success': False, 'error': 'Unknown request type'}
                    
                    response_queue.put(result)
                    
                except Exception as e:
                    error_result = {'success': False, 'error': str(e)}
                    response_queue.put(error_result)
                    self._log_error("REQUEST_PROCESSING_ERROR", "", str(e))
                
                finally:
                    self.is_processing = False
                    self.request_queue.task_done()
                    
            except:
                # Timeout or other error - continue processing
                continue
    
    def _queue_request(self, request_type: str, data: dict = None) -> dict:
        """Queue a request for sequential processing"""
        response_queue = Queue()
        request_data = {
            'type': request_type,
            'data': data,
            'response_queue': response_queue
        }
        
        self.request_queue.put(request_data)
        
        # Wait for response (with timeout)
        try:
            return response_queue.get(timeout=300)  # 5 minute timeout
        except:
            return {'success': False, 'error': 'Request timeout'}
    
    def _log_error(self, error_type: str, ticker: str, error_message: str, trade_data: dict = None):
        """Log errors to in-memory storage"""
        error_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": error_type,
            "ticker": ticker,
            "error_message": error_message,
            "trade_data": trade_data
        }
        
        self.error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
        
        print(f"🚨 Error logged: {error_type} - {error_message}")
    
    def _retry_with_backoff(self, func, max_retries=3, initial_delay=1.0, backoff_factor=2.0, error_name="OPERATION"):
        """Retry a function with exponential backoff
        
        Args:
            func: Function to retry (should return tuple of (success: bool, result: Any))
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for delay after each retry
            error_name: Name for logging purposes
        
        Returns:
            Tuple of (success: bool, result: Any)
        """
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries):
            try:
                success, result = func()
                if success:
                    if attempt > 0:
                        print(f"   ✅ {error_name} succeeded on attempt {attempt + 1}")
                    return True, result
                else:
                    last_error = result
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ {error_name} failed (attempt {attempt + 1}/{max_retries}): {result}")
                        print(f"   ⏳ Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    print(f"   ⚠️ {error_name} exception (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"   ⏳ Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
        
        print(f"   ❌ {error_name} failed after {max_retries} attempts: {last_error}")
        return False, last_error
    
    def _connect_to_ib(self, ticker: str = "UNKNOWN") -> bool:
        """Connect to IB Web API with retry logic for contract lookup"""
        try:
            self.ib_api = IBWebAPI()
            
            # Test connection
            if not self.ib_api.is_connected():
                error_msg = "Not authenticated with IBKR Web API. Please authenticate first."
                self._log_error("CONNECTION_FAILED", ticker, error_msg)
                print(f"❌ {error_msg}")
                return False
            
            # Test basic functionality
            test_response = self.ib_api.session.get(f"{self.ib_api.base_url}/iserver/accounts")
            if test_response.status_code != 200:
                error_msg = f"IBKR Web API test failed: {test_response.status_code}"
                self._log_error("CONNECTION_TEST_FAILED", ticker, error_msg)
                print(f"❌ {error_msg}")
                return False
            
            # Test account access
            print("✅ Testing account access...")
            self.ib_api.get_accounts()

            # Test contract lookup with retry logic
            print(f"✅ Testing contract lookup for {ticker}...")
            
            def lookup_contract():
                conid = self.ib_api.get_contract_id(ticker)
                if conid:
                    return True, conid
                else:
                    return False, "Contract ID not found"
            
            success, result = self._retry_with_backoff(
                lookup_contract,
                max_retries=3,
                initial_delay=1.0,
                backoff_factor=2.0,
                error_name=f"Contract lookup for {ticker}"
            )
            
            if not success:
                error_msg = f"Could not find contract ID for {ticker} after retries: {result}"
                self._log_error("CONTRACT_LOOKUP_FAILED", ticker, error_msg)
                print(f"❌ {error_msg}")
                return False
            
            print(f"✅ Found contract ID for {ticker}: {result}")
            print("✅ Connected to IBKR Web API")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to IBKR Web API: {str(e)}"
            print(f"❌ {error_msg}")
            self._log_error("CONNECTION_FAILED", ticker, error_msg)
            return False
    
    def _disconnect_from_ib(self):
        """Disconnect from IBKR Web API"""
        self.ib_api = None
        print("✅ Disconnected from IBKR Web API")
        
        self.ib_wrapper = None
        self.ib_client = None

    def test_ib_connection(self, sample_symbol: str = "SPY") -> dict:
        """Lightweight connectivity check for the IBKR Web API."""
        symbol = (sample_symbol or "SPY").upper()
        checked_at = datetime.utcnow().isoformat() + "Z"
        try:
            ib_api = IBWebAPI()
            
            if not ib_api.is_connected():
                return {
                    'success': False,
                    'stage': 'auth',
                    'message': (
                        'IBKR Web API is not authenticated. Ensure the ibeam container is running and logged in. '
                        'If login is stuck on 2FA, configure an IBeam 2FA handler (e.g. IBEAM_TWO_FA_HANDLER=EXTERNAL_REQUEST) '
                        'or complete the approval in IBKR Mobile. '
                        'Expected base URL: ' + ib_api.base_url
                    ),
                    'checked_at': checked_at,
                    'sample_symbol': symbol
                }
            
            accounts_response = ib_api.get_accounts()
            status_code = getattr(accounts_response, 'status_code', None)
            if not accounts_response or status_code != 200:
                return {
                    'success': False,
                    'stage': 'accounts',
                    'message': f'Unable to query IBKR accounts (status {status_code}).',
                    'checked_at': checked_at,
                    'sample_symbol': symbol
                }
            
            conid = ib_api.get_contract_id(symbol)
            if not conid:
                return {
                    'success': False,
                    'stage': 'contract_lookup',
                    'message': f'Connected but could not look up contract ID for {symbol}.',
                    'checked_at': checked_at,
                    'sample_symbol': symbol
                }
            
            return {
                'success': True,
                'stage': 'ready',
                'message': 'IBKR Web API responded successfully.',
                'checked_at': checked_at,
                'sample_symbol': symbol,
                'sample_conid': conid
            }
        
        except Exception as e:
            error_msg = f"IB status check failed: {str(e)}"
            self._log_error("IB_STATUS_CHECK_FAILED", symbol, error_msg)
            return {
                'success': False,
                'stage': 'exception',
                'message': error_msg,
                'checked_at': checked_at,
                'sample_symbol': symbol
            }

    def get_symbol_algos(self, symbol: str, add_description: bool = True, add_params: bool = True, algos: Optional[str] = None) -> dict:
        """Get available IBKR algos for a symbol/contract."""
        normalized_symbol = (symbol or "").upper().strip()
        if not normalized_symbol:
            return {'success': False, 'error': 'symbol is required'}

        try:
            ib_api = IBWebAPI()
            if not ib_api.is_connected():
                return {
                    'success': False,
                    'error': 'IBKR Web API is not authenticated',
                    'symbol': normalized_symbol,
                }

            conid = ib_api.get_contract_id(normalized_symbol)
            if not conid:
                return {
                    'success': False,
                    'error': f'Could not resolve conid for {normalized_symbol}',
                    'symbol': normalized_symbol,
                }

            algo_list = ib_api.get_contract_algos(
                conid,
                add_description=add_description,
                add_params=add_params,
                algos=algos,
            )
            if algo_list is None:
                return {
                    'success': False,
                    'error': f'Failed to fetch algos for {normalized_symbol}',
                    'symbol': normalized_symbol,
                    'conid': conid,
                }

            return {
                'success': True,
                'symbol': normalized_symbol,
                'conid': conid,
                'algos': algo_list,
                'algo_ids': [a.get('id') for a in algo_list if isinstance(a, dict) and a.get('id')],
                'adaptive_available': any(isinstance(a, dict) and a.get('id') == 'Adaptive' for a in algo_list),
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': normalized_symbol,
            }
        
    def _execute_order(self, symbol: str, side: str, quantity: float, order_type: str = "MKT", price: float = None, stop_price: float = None, tif: str = "DAY", adaptive_priority: Optional[str] = None) -> dict:
        try:
            conid = self.ib_api.get_contract_id(symbol)
            if not conid:
                return {'success': False, 'error': f'Could not find contract for {symbol}'}

            # Validate quantity
            if quantity < 0.1:
                print(f"   ⚠️ Warning: Order quantity {quantity} is very small and may not be supported for {symbol}")
                return {'success': False, 'error': f'Quantity {quantity} too small for {symbol}'}

            if order_type == "IBALGO":
                if abs(quantity - round(quantity)) > 1e-9:
                    return {'success': False, 'error': f'IBALGO requires whole-share quantity. Received {quantity}'}
                quantity = int(round(quantity))

                # Preflight: ensure Adaptive is explicitly available for this conid.
                available_algos = self.ib_api.get_contract_algos(conid)
                if available_algos is None:
                    return {'success': False, 'error': f'Unable to fetch available algos for {symbol} (conid {conid}). Aborting IBALGO order.'}

                available_ids = [str(a.get('id')) for a in available_algos if isinstance(a, dict) and a.get('id')]
                if 'Adaptive' not in available_ids:
                    return {
                        'success': False,
                        'error': (
                            f"Adaptive algo not available for {symbol} (conid {conid}). "
                            f"Available algos: {', '.join(available_ids) if available_ids else 'none'}"
                        )
                    }

                print(f"   ✅ Adaptive available for {symbol} (conid {conid}). Available algos: {available_ids}")

            order_data = {
                "orderType": "MKT" if order_type == "IBALGO" else order_type,
                "side": side.upper(),
                "quantity": quantity
            }

            if order_type == "IBALGO":
                priority = adaptive_priority or "Urgent"
                order_data["algoStrategy"] = "Adaptive"
                order_data["algoParams"] = [{"tag": "adaptivePriority", "value": priority}]
                # Also include alternate field names used by some IBKR docs/builds.
                order_data["strategy"] = "Adaptive"
                order_data["strategyParameters"] = {"adaptivePriority": priority}
            if price:
                order_data["price"] = price
            if stop_price:
                order_data["auxPrice"] = stop_price
                if order_type == "STP":  # For stop orders, set price equal to auxPrice
                    order_data["price"] = stop_price

            order_data["tif"] = tif

            response = self.ib_api.place_order(conid, order_data)
            print(f"   📤 Order request sent. Status: {response.status_code}")
            print(f"   📋 Order data: {order_data}")
            print(f"   📋 Contract ID: {conid}")
            print(f"   📥 Response headers: {dict(response.headers)}")
            print(f"   📥 Response body: {response.text}")

            if response.status_code != 200:
                error_details = f'Status: {response.status_code}, Response: {response.text}'
                print(f"   ❌ Order failed - {error_details}")
                return {'success': False, 'error': error_details}

            result = response.json()
            print(f"   ✅ Order response: {result}")

            # Handle multiple confirmations
            max_confirmations = 3
            confirmation_count = 0
            current_result = result

            while isinstance(current_result, list) and len(current_result) > 0 and 'id' in current_result[0] and confirmation_count < max_confirmations:
                confirmation_id = current_result[0]['id']
                message_ids = current_result[0].get('messageIds', []) or []
                messages = current_result[0].get('message', []) or []
                message_blob = ' '.join(str(m) for m in messages).lower()

                if order_type == "IBALGO":
                    market_warning_ids = {'o10151', 'o10153'}
                    has_market_warning = any(str(mid).lower() in market_warning_ids for mid in message_ids) or ('market order confirmation' in message_blob)
                    strict_decline_market_warning = str(os.getenv('IBALGO_DECLINE_MARKET_WARNING', '0')).lower() in {'1', 'true', 'yes'}
                    if has_market_warning and strict_decline_market_warning:
                        print("   🛑 IBALGO strict guard: declining market-order confirmation to avoid plain MKT execution.")
                        decline_response = self.ib_api.session.post(
                            f"{self.ib_api.base_url}/iserver/reply/{confirmation_id}",
                            json={"confirmed": False},
                            timeout=30
                        )
                        print(f"   📥 Decline response status: {decline_response.status_code}, Body: {decline_response.text}")
                        return {
                            'success': False,
                            'error': 'IBALGO strict guard declined market-order confirmation; order not submitted.'
                        }
                    if has_market_warning and not strict_decline_market_warning:
                        print("   ⚠️ IBALGO market warning received; continuing because IBALGO_DECLINE_MARKET_WARNING=0")

                print(f"   📩 Confirmation required. Sending reply to ID: {confirmation_id}")
                reply_response = self.ib_api.session.post(
                    f"{self.ib_api.base_url}/iserver/reply/{confirmation_id}",
                    json={"confirmed": True},
                    timeout=30
                )
                print(f"   📥 Reply response status: {reply_response.status_code}, Headers: {dict(reply_response.headers)}, Body: {reply_response.text}")
                if reply_response.status_code != 200:
                    return {
                        'success': False,
                        'error': f'Confirmation failed: Status {reply_response.status_code}, Response: {reply_response.text}'
                    }
                current_result = reply_response.json()
                print(f"   ✅ Confirmation response: {current_result}")
                confirmation_count += 1

            # Extract order_id from final response
            order_id = None
            if isinstance(current_result, list) and current_result:
                order_id = current_result[0].get('order_id') or current_result[0].get('id')
            elif isinstance(current_result, dict):
                order_id = current_result.get('order_id') or current_result.get('id')

            if order_id:
                print(f"   📤 ORDER SUBMITTED (ID: {order_id}, side={side.upper()}, type={order_data.get('orderType')})")
                # Immediate status check
                time.sleep(1)  # Short delay to allow order registration
                immediate_order = self.ib_api.find_order_by_id(order_id)
                print(f"   📋 Immediate order lookup: {immediate_order}")
                ibalgo_ack = True
                ibalgo_status = None
                if order_type == "IBALGO":
                    expected_priority = adaptive_priority or "Urgent"
                    ibalgo_status = self._verify_ibalgo_ack(order_id, expected_priority)
                    ibalgo_ack = ibalgo_status.get('acknowledged', False)
                    if ibalgo_ack:
                        print(f"   ✅ IBALGO acknowledged by IBKR (priority={ibalgo_status.get('priority', 'unknown')})")
                    else:
                        print("   ⚠️ IBALGO not confirmed in order status. It may have been accepted as plain MKT or status fields may be delayed.")
                        print(f"   ⚠️ IBALGO diagnostic: {ibalgo_status}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'message': 'Order confirmed and placed',
                    'ibalgo_acknowledged': ibalgo_ack,
                    'ibalgo_status': ibalgo_status
                }
            else:
                error_msg = f'Confirmation succeeded but no order_id returned: {current_result}'
                print(f"   ❌ BUY ORDER SUBMISSION FAILED: {error_msg}")
                return {'success': False, 'error': error_msg}

        except Exception as e:
            error_msg = f"Order execution failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {'success': False, 'error': error_msg}

    def _verify_ibalgo_ack(self, order_id: str, expected_priority: str) -> Dict[str, Any]:
        """Best-effort verification that IBKR status reflects Adaptive algo settings."""
        try:
            response = self.ib_api.get_order_status()
            if response.status_code != 200:
                return {
                    'acknowledged': False,
                    'reason': f'status_{response.status_code}',
                    'status_text': response.text,
                }

            payload = response.json()
            order_info = None
            if isinstance(payload, list):
                order_info = payload[0] if payload else None
            elif isinstance(payload, dict):
                if 'orders' in payload and isinstance(payload.get('orders'), list):
                    match = None
                    for o in payload.get('orders', []):
                        if str(o.get('orderId') or o.get('order_id') or o.get('id')) == str(order_id):
                            match = o
                            break
                    order_info = match or (payload.get('orders')[0] if payload.get('orders') else None)
                else:
                    order_info = payload

            if not isinstance(order_info, dict):
                return {
                    'acknowledged': False,
                    'reason': 'missing_order_info',
                    'payload': payload,
                }

            # Ensure we are checking the intended order id.
            oid = order_info.get('orderId') or order_info.get('order_id') or order_info.get('id')
            if str(oid) != str(order_id):
                exact = self.ib_api.find_order_by_id(order_id)
                if isinstance(exact, dict):
                    order_info = exact

            strategy = (
                order_info.get('algoStrategy')
                or order_info.get('algo_strategy')
                or order_info.get('strategy')
                or order_info.get('strategy')
                or ''
            )

            algo_params = order_info.get('algoParams') or order_info.get('algo_params') or []
            actual_priority = None
            if isinstance(algo_params, list):
                for item in algo_params:
                    if isinstance(item, dict) and str(item.get('tag', '')).lower() == 'adaptivepriority':
                        actual_priority = item.get('value')
                        break
            elif isinstance(algo_params, dict):
                for k, v in algo_params.items():
                    if str(k).lower() == 'adaptivepriority':
                        actual_priority = v
                        break

            order_desc = str(order_info.get('orderDesc') or order_info.get('order_desc') or '')
            adaptive_in_desc = 'adaptive' in order_desc.lower()

            acknowledged = (str(strategy).lower() == 'adaptive' or adaptive_in_desc) and (
                actual_priority is None or str(actual_priority).lower() == str(expected_priority).lower()
            )

            return {
                'acknowledged': acknowledged,
                'strategy': strategy,
                'priority': actual_priority,
                'expected_priority': expected_priority,
                'order_type': order_info.get('orderType') or order_info.get('order_type'),
                'status': order_info.get('status') or order_info.get('orderStatus'),
                'order_desc': order_desc,
            }
        except Exception as e:
            return {
                'acknowledged': False,
                'reason': 'exception',
                'error': str(e),
            }

    def _wait_for_order_fill_webapi(self, order_id: str, expected_shares: float, timeout: int = 7) -> dict:
        """Wait for an order to fill, handling partial fills and cancellation with proper account context."""
        print(f"   ⏳ Waiting for order {order_id} to fill {expected_shares} shares...")
        
        start_time = time.time()
        last_filled = 0.0
        no_progress_time = 0
        max_retries = 3
        
        # Initial delay to allow order registration
        time.sleep(2)
        
        while time.time() - start_time < timeout:
            for attempt in range(max_retries):
                try:
                    response = self.ib_api.get_order_status()
                    print(f"   📋 Full order status response (attempt {attempt + 1}): {response.text}")

                    order_info = None
                    if response.status_code == 200:
                        order_info = self.ib_api.find_order_by_id(order_id)
                    
                    # Check trades endpoint as a fallback
                    if not order_info:
                        print(f"   ⚠️ Order {order_id} not found in order list (attempt {attempt + 1})")
                        print(f"   ⚠️ Checking trades for order {order_id}...")
                        trades_response = self.ib_api.session.get(f"{self.ib_api.base_url}/iserver/account/trades", timeout=30)
                        if trades_response.status_code == 200:
                            trades = trades_response.json()
                            for trade in trades:
                                if str(trade.get('order_id')) == str(order_id):
                                    print(f"   ✅ Found trade: {trade}")
                                    return {
                                        'success': True,
                                        'filled_shares': float(trade.get('executed_qty', 0)),
                                        'remaining_shares': 0,
                                        'avg_price': float(trade.get('avg_price', 0)),
                                        'status': 'FILLED',
                                        'cancelled': False
                                    }
                        print(f"   ⚠️ Order {order_id} not found in trades")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        break
                    
                    # Process order information
                    status = order_info.get('status', order_info.get('orderStatus', 'Unknown')).upper()
                    filled_qty = float(order_info.get('filledQuantity', order_info.get('filled', 0)))
                    remaining_qty = float(order_info.get('remainingQuantity', order_info.get('remaining', expected_shares - filled_qty)))
                    avg_price = float(order_info.get('avgPrice', order_info.get('avgFillPrice', 0)))
                    
                    if filled_qty > last_filled:
                        last_filled = filled_qty
                        no_progress_time = 0
                        print(f"   📈 Progress: {filled_qty}/{expected_shares} shares filled at avg ${avg_price}")
                    
                    if status in ['FILLED', 'COMPLETE']:
                        print(f"   ✅ Order {order_id} FULLY FILLED: {filled_qty} shares at ${avg_price}")
                        return {
                            'success': True,
                            'filled_shares': filled_qty,
                            'remaining_shares': 0,
                            'avg_price': avg_price,
                            'status': status,
                            'cancelled': False
                        }
                    elif status in ['CANCELLED', 'CANCELED']:
                        print(f"   ❌ Order {order_id} CANCELLED: {filled_qty} shares filled, {remaining_qty} remaining")
                        return {
                            'success': filled_qty > 0,
                            'filled_shares': filled_qty,
                            'remaining_shares': remaining_qty,
                            'avg_price': avg_price,
                            'status': status,
                            'cancelled': True
                        }
                    elif status in ['PRESUBMITTED']:
                        print(f"   ⏳ Order {order_id} in PreSubmitted state, waiting for status change...")
                        time.sleep(3)
                        continue
                    elif filled_qty > 0 and filled_qty < expected_shares:
                        if status in ['PARTIALLYFILLED', 'SUBMITTED', 'PENDING']:
                            no_progress_time += 1
                            print(f"   ⏳ Partial fill: {filled_qty}/{expected_shares} shares. Waiting for more...")
                            time.sleep(1)
                            continue
                    
                    time.sleep(1)
                    no_progress_time += 1
                    break
                except Exception as e:
                    print(f"   ⚠️ Error checking order status: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    time.sleep(1)
                    no_progress_time += 1
                    break
        
        # Timeout reached - final check
        print(f"   ⏰ Order {order_id} timeout. Checking final status...")
        final_filled = 0.0
        final_remaining = expected_shares
        final_avg = 0.0
        final_status = "Timeout"
        cancelled = False
        try:
            response = self.ib_api.get_order_status()  # Full order list
            if response.status_code == 200:
                order_info = self.ib_api.find_order_by_id(order_id)
                if order_info:
                    final_filled = float(order_info.get('filledQuantity', 0))
                    final_remaining = float(order_info.get('remainingQuantity', expected_shares - final_filled))
                    final_avg = float(order_info.get('avgPrice', 0))
                    final_status = order_info.get('status')
                    print(f"   📋 Final status: {final_status}, Filled: {final_filled}, Remaining: {final_remaining}")
                    if final_remaining > 0:
                        print(f"   📤 Cancelling remaining {final_remaining} shares...")
                        cancel_response = self.ib_api.cancel_order(order_id)
                        if isinstance(cancel_response, dict):
                            # Handle error case from cancel_order
                            if cancel_response.get('status_code') in [200, 201]:
                                cancelled = True
                                final_remaining = 0.0
                                print("   ✅ Cancelled remaining")
                            else:
                                print(f"   ❌ Cancel failed: {cancel_response.get('text', 'Unknown error')}")
                        else:
                            # Handle standard response object
                            if cancel_response.status_code in [200, 201]:
                                cancelled = True
                                final_remaining = 0.0
                                print("   ✅ Cancelled remaining")
                            else:
                                print(f"   ❌ Cancel failed: {cancel_response.status_code}, {cancel_response.text}")
            if final_filled == 0:  # Fallback to trades if no filled found
                print(f"   ⚠️ Checking trades for final status of order {order_id}...")
                trades_response = self.ib_api.session.get(f"{self.ib_api.base_url}/iserver/account/trades", timeout=30)
                if trades_response.status_code == 200:
                    trades = trades_response.json()
                    exec_qty_total = 0.0
                    price_total = 0.0
                    for trade in trades:
                        if str(trade.get('order_id')) == str(order_id):
                            exec_qty = float(trade.get('executed_qty', 0))
                            exec_qty_total += exec_qty
                            price_total += float(trade.get('avg_price', 0)) * exec_qty
                    if exec_qty_total > 0:
                        final_filled = exec_qty_total
                        final_avg = price_total / exec_qty_total
                        final_remaining = 0.0
                        final_status = "Filled"
                        print(f"   📋 Found filled from trades: {final_filled} shares")
        except Exception as e:
            print(f"   ❌ Error in final check: {str(e)}")
            self._log_error("FINAL_STATUS_CHECK_FAILED", "UNKNOWN", str(e), order_id=order_id)
        
        return {
            'success': final_filled > 0,
            'filled_shares': final_filled,
            'remaining_shares': final_remaining,
            'avg_price': final_avg,
            'status': final_status,
            'cancelled': cancelled
        }
    
    def _execute_buy_order(self, trade: Trade) -> dict:
        """Execute buy order via Web API with partial order handling"""
        print(f"\n🔵 EXECUTING BUY ORDER:")
        print(f"   Ticker: {trade.ticker}")
        print(f"   Shares: {trade.shares}")
        print(f"   Risk Amount: ${trade.risk_amount}")
        print(f"   Type: {trade.order_type}")
        if trade.order_type == "IBALGO":
            print(f"   Adaptive Priority: {trade.adaptive_priority or 'Urgent'}")
        print(f"   Timeout: {trade.timeout_seconds}s")
        
        try:
            result = self._execute_order(
                trade.ticker,
                "BUY",
                trade.shares,
                order_type=trade.order_type,
                adaptive_priority=trade.adaptive_priority
            )
            
            if not result['success']:
                error_msg = f"Buy order submission failed: {result['error']}"
                print(f"   ❌ BUY ORDER SUBMISSION FAILED: {result['error']}")
                self._log_error("BUY_ORDER_FAILED", trade.ticker, error_msg)
                return {
                    'success': False,
                    'filled_shares': 0,
                    'avg_price': 0,
                    'full_fill': False
                }
            
            order_id = result['order_id']
            print(f"   📤 BUY ORDER SUBMITTED (Order ID: {order_id})")

            require_ibalgo_ack = str(os.getenv('IBALGO_REQUIRE_ACK', '0')).lower() not in {'0', 'false', 'no'}
            if trade.order_type == "IBALGO" and require_ibalgo_ack and not result.get('ibalgo_acknowledged', False):
                warn_msg = (
                    f"IBALGO not acknowledged by IBKR for {trade.ticker} (order {order_id}). "
                    f"Attempting immediate cancel to avoid plain MKT execution."
                )
                print(f"   ⚠️ {warn_msg}")
                self._log_error("IBALGO_NOT_ACKNOWLEDGED", trade.ticker, warn_msg, result.get('ibalgo_status'))

                try:
                    cancel_response = self.ib_api.cancel_order(order_id)
                    if isinstance(cancel_response, dict):
                        print(f"   📤 Cancel response (dict): {cancel_response}")
                    else:
                        print(f"   📤 Cancel response: {cancel_response.status_code} {cancel_response.text}")
                except Exception as cancel_err:
                    print(f"   ⚠️ Cancel attempt raised: {cancel_err}")

                # If anything filled before cancel, continue and protect the position with stops.
                fill_after_cancel = self._wait_for_order_fill_webapi(order_id, trade.shares, timeout=1)
                if fill_after_cancel.get('filled_shares', 0) > 0:
                    filled_shares = fill_after_cancel['filled_shares']
                    avg_price = fill_after_cancel['avg_price']
                    full_fill = filled_shares == trade.shares
                    print(f"   ⚠️ IBALGO not acknowledged, but {filled_shares} shares filled before cancel.")
                    return {
                        'success': True,
                        'filled_shares': filled_shares,
                        'avg_price': avg_price,
                        'full_fill': full_fill
                    }

                print("   ✅ Unacknowledged IBALGO was cancelled before fill.")
                return {
                    'success': False,
                    'filled_shares': 0,
                    'avg_price': 0,
                    'full_fill': False
                }
            
            # Wait for order to fill with partial handling
            fill_result = self._wait_for_order_fill_webapi(order_id, trade.shares, timeout=max(1, int(trade.timeout_seconds)))
            
            if fill_result['filled_shares'] > 0:
                filled_shares = fill_result['filled_shares']
                avg_price = fill_result['avg_price']
                full_fill = filled_shares == trade.shares
                print(f"   ✅ BUY ORDER {'FULLY ' if full_fill else 'PARTIALLY '}COMPLETED: {filled_shares} shares at ${avg_price}")
                return {
                    'success': True,
                    'filled_shares': filled_shares,
                    'avg_price': avg_price,
                    'full_fill': full_fill
                }
            else:
                error_msg = f"Buy order {order_id} failed to fill any shares"
                print(f"   ❌ BUY ORDER FAILED: {error_msg}")
                self._log_error("BUY_ORDER_NO_FILL", trade.ticker, error_msg)
                return {
                    'success': False,
                    'filled_shares': 0,
                    'avg_price': 0,
                    'full_fill': False
                }
                
        except Exception as e:
            error_msg = f"Buy order failed: {str(e)}"
            print(f"   ❌ BUY ORDER FAILED: {str(e)}")
            self._log_error("BUY_ORDER_FAILED", trade.ticker, error_msg)
            return {
                'success': False,
                'filled_shares': 0,
                'avg_price': 0,
                'full_fill': False
            }
    
    def _execute_sell_stop_orders(self, trade: Trade, actual_shares_bought: float, avg_fill_price: float | None = None):
        """Execute sell stop orders based on actual shares bought with proper scaling.

        For stops defined with percent_below_fill, avg_fill_price is required (will use provided avg_fill_price
        or attempt to look up last trade fill price if None).
        """
        print(f"\n🔴 SETTING SELL STOP ORDERS for {actual_shares_bought} shares:")
        
        failed_stops = []  # Track failed stops for critical alerting
        
        if actual_shares_bought == 0:
            print("   ❌ No shares bought - skipping sell stop orders")
            return
        
        total_planned_shares = trade.shares
        scale_factor = float(actual_shares_bought) / total_planned_shares
        
        # Get contract ID for tick size validation
        conid = self.ib_api.get_contract_id(trade.ticker)
        if not conid:
            print(f"   ❌ Could not find contract for {trade.ticker}")
            self._log_error("CONTRACT_NOT_FOUND", trade.ticker, "Could not find contract ID")
            return
        
        # Get contract details for tick size
        contract_details = self.ib_api.get_contract_details(conid)
        price_increment = 0.01  # Default tick size
        if contract_details:
            price_increment = float(contract_details.get('priceIncrement', 0.01))
            print(f"   📏 Price increment (tick size): ${price_increment}")
        
        try:
            for i, stop in enumerate(trade.sell_stops, 1):
                try:
                    # Scale the shares but keep fractional precision for Web API
                    scaled_shares = stop.shares * scale_factor
                    
                    if scaled_shares < 0.001:  # Minimum fractional share
                        print(f"   ⚠️ Stop {i}: Skipping (scaled to {scaled_shares:.3f} shares - too small)")
                        continue
                    
                    # Determine raw stop price (fixed or percentage)
                    if stop.price is not None:
                        raw_stop_price = stop.price
                    else:
                        if avg_fill_price is None:
                            print(f"   ⚠️ Stop {i}: percent_below_fill provided but avg_fill_price unknown - skipping")
                            continue
                        raw_stop_price = avg_fill_price * (1 - stop.percent_below_fill / 100.0)
                        print(f"   📐 Stop {i}: {stop.percent_below_fill}% below fill (${avg_fill_price:.2f}) => raw ${raw_stop_price:.4f}")

                    # Adjust to tick size
                    adjusted_stop_price = round(raw_stop_price / price_increment) * price_increment
                    if abs(adjusted_stop_price - raw_stop_price) > 0.0001:
                        print(f"   🔧 Adjusted stop price for {trade.ticker} from ${raw_stop_price:.4f} to ${adjusted_stop_price:.4f}")
                    
                    result = self._execute_order(
                        trade.ticker, 
                        "SELL", 
                        scaled_shares,
                        "STP", 
                        stop_price=adjusted_stop_price,
                        tif="GTC"
                    )
                    
                    if result['success']:
                        print(f"   Stop {i}: {scaled_shares:.3f} shares at ${adjusted_stop_price} - Order ID: {result.get('order_id', 'N/A')}")
                    else:
                        print(f"   ❌ Stop {i} FAILED: {result['error']}")
                        self._log_error("SELL_STOP_ORDER_FAILED", trade.ticker, result['error'])
                        failed_stops.append((i, scaled_shares, adjusted_stop_price, result['error']))
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    error_msg = f"Sell stop order {i} failed: {str(e)}"
                    print(f"   ❌ SELL STOP ORDER {i} FAILED: {str(e)}")
                    self._log_error("SELL_STOP_ORDER_FAILED", trade.ticker, error_msg)
                    failed_stops.append((i, stop.shares * scale_factor, stop.price or 'percent-based', str(e)))
                    continue
            
            # CRITICAL: Alert if any stops failed - position may be unprotected
            if failed_stops:
                total_unprotected = sum(f[1] for f in failed_stops)
                critical_msg = (f"🚨 CRITICAL: {len(failed_stops)} stop order(s) FAILED for {trade.ticker}! "
                               f"{total_unprotected:.3f} shares may be UNPROTECTED. Manual intervention required!")
                print(f"\n{'='*60}")
                print(critical_msg)
                for stop_num, shares, price, err in failed_stops:
                    print(f"   - Stop {stop_num}: {shares:.3f} shares @ ${price} - {err}")
                print(f"{'='*60}\n")
                self._log_error("CRITICAL_STOPS_FAILED", trade.ticker, critical_msg, 
                               {'failed_stops': failed_stops, 'total_unprotected_shares': total_unprotected})
            else:
                print(f"   ✅ SELL STOP ORDERS PLACED")
            
        except Exception as e:
            error_msg = f"Sell stop orders failed: {str(e)}"
            print(f"   ❌ SELL STOP ORDERS FAILED: {str(e)}")
            self._log_error("SELL_STOP_ORDERS_FAILED", trade.ticker, error_msg)
            raise
    
    def _find_trade_by_criteria(self, ticker: str, lower_price: float, higher_price: float) -> Optional[Dict[str, Any]]:
        """Find trade by criteria in database"""
        return db_find_trade(ticker, lower_price, higher_price)
    
    def _validate_trade(self, trade: Trade) -> tuple[bool, str]:
        """Validate trade data and return detailed error message"""
        normalized_order_type = (trade.order_type or "MKT").upper()
        if normalized_order_type not in {"MKT", "IBALGO"}:
            return False, f"Invalid order_type: {trade.order_type}. Must be MKT or IBALGO"
        trade.order_type = normalized_order_type

        if trade.order_type == "IBALGO":
            valid_priorities = {"Patient", "Normal", "Urgent"}
            priority = trade.adaptive_priority or "Urgent"
            if priority not in valid_priorities:
                return False, f"Invalid adaptive_priority: {priority}. Must be Patient, Normal, or Urgent"
            trade.adaptive_priority = priority
            if abs(trade.shares - round(trade.shares)) > 1e-9:
                return False, f"IBALGO trades require whole shares. Received {trade.shares}"
        else:
            trade.adaptive_priority = None

        if trade.timeout_seconds is None:
            trade.timeout_seconds = 30 if trade.order_type == "IBALGO" else 5
        try:
            trade.timeout_seconds = int(trade.timeout_seconds)
        except Exception:
            return False, f"Invalid timeout_seconds: {trade.timeout_seconds}"
        if trade.timeout_seconds <= 0:
            return False, f"timeout_seconds must be > 0. Received {trade.timeout_seconds}"

        total_stop_shares = 0.0
        for idx, stop in enumerate(trade.sell_stops, 1):
            total_stop_shares += stop.shares
            # Each stop must define exactly one of price or percent_below_fill
            if (stop.price is None and stop.percent_below_fill is None) or (stop.price is not None and stop.percent_below_fill is not None):
                return False, f"Sell stop {idx} must have exactly one of price or percent_below_fill"
            if stop.percent_below_fill is not None:
                if stop.percent_below_fill <= 0 or stop.percent_below_fill >= 100:
                    return False, f"Sell stop {idx} percent_below_fill must be between 0 and 100"
            if stop.price is not None and stop.price <= 0:
                return False, f"Sell stop {idx} price must be > 0"
            if stop.shares <= 0:
                return False, f"Sell stop {idx} shares must be > 0"
        if abs(total_stop_shares - trade.shares) > 0.001:
            error_msg = f"Sell stop shares ({total_stop_shares}) don't match total shares ({trade.shares})"
            print(f"ERROR: {error_msg}")
            return False, error_msg
        
        if trade.shares <= 0:
            error_msg = f"Invalid shares amount: {trade.shares}"
            return False, error_msg
        
        if len(trade.sell_stops) == 0:
            error_msg = "No sell stop orders defined"
            return False, error_msg
        
        return True, ""
    
    def _execute_trade_internal(self, data: dict) -> dict:
        """Internal method to execute trade with Delete-Verify-Then-Send safety.
        
        Safety Flow:
        1. Find trade in DB
        2. Delete from DB
        3. Verify deletion (trade gone)
        4. Only then send order to IBKR
        
        This ensures a trade can NEVER be executed twice.
        """
        ticker = data['ticker']
        lower_price = data['lower_price']
        higher_price = data['higher_price']
        
        print(f"\n📊 Looking for trade: {ticker} (${lower_price} - ${higher_price})...")
        
        # Step 1: Find the trade in database
        trade_data = db_find_trade(ticker, lower_price, higher_price)
        
        if trade_data is None:
            error_msg = f"No trade found for {ticker} with price range ${lower_price} - ${higher_price}"
            self._log_error("TRADE_NOT_FOUND", ticker, error_msg)
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
        
        trade_id = trade_data['trade_id']
        print(f"✅ Found trade for {ticker} (ID: {trade_id})")
        
        # Convert DB data to Trade object for validation and execution
        sell_stops = []
        for stop_data in trade_data['sell_stops']:
            sell_stops.append(SellStopOrder(
                price=stop_data.get('price'),
                shares=stop_data['shares'],
                percent_below_fill=stop_data.get('percent_below_fill')
            ))
        
        trade = Trade(
            ticker=trade_data['ticker'],
            shares=trade_data['shares'],
            risk_amount=trade_data['risk_amount'],
            lower_price_range=trade_data['lower_price_range'],
            higher_price_range=trade_data['higher_price_range'],
            sell_stops=sell_stops,
            order_type=trade_data.get('order_type', 'MKT'),
            adaptive_priority=trade_data.get('adaptive_priority'),
            timeout_seconds=trade_data.get('timeout_seconds', 5),
            trade_id=trade_id
        )

        def _return_risk_only(reason: str):
            """Restore risk pool when execution aborts before any fill. Trade is NOT requeued — add it back manually if needed."""
            print(f"🔄 Restoring risk (trade NOT requeued) due to: {reason}")

            # Restore risk pool only.
            self.available_risk += trade.risk_amount
            if self.available_risk < 0 and self.available_risk > -1e-6:
                self.available_risk = 0.0
            save_risk_amount(self.available_risk)
            print(f"✅ Risk restored: +${trade.risk_amount:.2f}. Available risk: ${self.available_risk:.2f}")
            print(f"ℹ️ Trade {trade.trade_id} ({trade.ticker}) was NOT requeued. Add it back manually if you want to retry.")
        
        # Validate trade before deletion
        is_valid, error_msg = self._validate_trade(trade)
        if not is_valid:
            print(f"❌ {error_msg}. Removing invalid trade.")
            self._log_error("TRADE_VALIDATION_FAILED", ticker, error_msg)
            db_delete_trade(trade_id)  # Remove invalid trade
            return {'success': False, 'error': error_msg}
        
        # RISK CHECK (before deletion - don't burn a trade if we can't afford it)
        if self.available_risk < trade.risk_amount - 1e-9:
            error_msg = (f"Insufficient available risk: have ${self.available_risk:.2f}, "
                         f"need ${trade.risk_amount:.2f} for trade {trade.ticker}")
            print(f"❌ {error_msg}")
            self._log_error("INSUFFICIENT_RISK", ticker, error_msg)
            # Do NOT remove trade; allow user to either increase risk or remove trade later
            return {'success': False, 'error': error_msg, 'available_risk': self.available_risk}
        
        # ============================================================
        # CRITICAL SAFETY: DELETE-VERIFY-THEN-SEND
        # ============================================================
        
        # Step 2: Delete trade from database
        print(f"🗑️ Deleting trade {trade_id} from database...")
        if not db_delete_trade(trade_id):
            error_msg = f"Failed to delete trade {trade_id} - may have already been executed"
            print(f"🛑 {error_msg}")
            self._log_error("DELETE_FAILED", ticker, error_msg)
            return {'success': False, 'error': error_msg}
        
        print(f"✅ Trade deleted from database")
        
        # Step 3: VERIFY deletion - trade must NOT exist anymore
        print(f"🔍 Verifying trade {trade_id} is gone from database...")
        if db_trade_exists(trade_id):
            # This should NEVER happen, but if it does, ABORT
            error_msg = f"CRITICAL: Trade {trade_id} still exists after deletion - ABORTING to prevent duplicate"
            print(f"🚨 {error_msg}")
            self._log_error("VERIFICATION_FAILED", ticker, error_msg)
            return {'success': False, 'error': error_msg}
        
        print(f"✅ Verified: Trade {trade_id} no longer exists in database")
        
        # Deduct risk now that trade is confirmed deleted
        self.available_risk -= trade.risk_amount
        if self.available_risk < 0 and self.available_risk > -1e-6:
            self.available_risk = 0.0
        save_risk_amount(self.available_risk)  # Persist immediately for crash safety
        print(f"✅ Risk deducted: -${trade.risk_amount:.2f}. New available risk: ${self.available_risk:.2f}")
        
        # ============================================================
        # Step 4: ONLY NOW send order to IBKR (trade is gone from DB)
        # ============================================================
        
        # Connect to IB
        if not self._connect_to_ib(ticker):
            error_msg = "Failed to connect to IB"
            self._log_error("CONNECTION_FAILED", ticker, error_msg)
            _return_risk_only("connect_failed")
            return {'success': False, 'error': error_msg, 'available_risk': self.available_risk}

        try:
            # Update last trade time
            self.last_trade_time = time.time()

            # Execute buy order
            buy_result = self._execute_buy_order(trade)
            
            if not buy_result['success']:
                error_msg = "Buy order failed completely"
                print(f"❌ {error_msg}")
                self._log_error("BUY_ORDER_COMPLETE_FAILURE", ticker, error_msg)
                _return_risk_only("buy_failed_before_fill")
                return {'success': False, 'error': error_msg}
            
            # Place sell stops
            self._execute_sell_stop_orders(trade, buy_result['filled_shares'], buy_result['avg_price'])
            
            risk_used = trade.risk_amount
            print(f"💰 Risk used: ${risk_used:.2f}")
            
            result = {
                'success': True,
                'ticker': trade.ticker,
                'filled_shares': buy_result['filled_shares'],
                'avg_price': buy_result['avg_price'],
                'full_fill': buy_result['full_fill'],
                'order_type': trade.order_type,
                'adaptive_priority': trade.adaptive_priority,
                'timeout_seconds': trade.timeout_seconds,
                'risk_used': risk_used,
                'available_risk': self.available_risk
            }
            
            print(f"\n🎉 Trade for {trade.ticker} completed!")
            return result
            
        except Exception as e:
            error_msg = f"Error processing trade: {str(e)}"
            print(f"❌ {error_msg}")
            self._log_error("TRADE_PROCESSING_ERROR", ticker, error_msg)
            return {'success': False, 'error': error_msg}
            
        finally:
            self._disconnect_from_ib()
    
    def _add_trade_internal(self, trade_data: dict) -> dict:
        """Internal method to add trade to database"""
        try:
            sell_stops_data = []
            sell_stops = []
            for stop in trade_data.get('sell_stops', []):
                price_val = stop.get('price')
                pct_val = stop.get('percent_below_fill')
                stop_obj = SellStopOrder(
                    price=float(price_val) if price_val is not None else None,
                    shares=float(stop['shares']),
                    percent_below_fill=float(pct_val) if pct_val is not None else None
                )
                sell_stops.append(stop_obj)
                # Also prepare for DB storage
                sell_stops_data.append({
                    'price': float(price_val) if price_val is not None else None,
                    'shares': float(stop['shares']),
                    'percent_below_fill': float(pct_val) if pct_val is not None else None
                })
            
            trade = Trade(
                ticker=trade_data['ticker'],
                shares=float(trade_data['shares']),
                risk_amount=float(trade_data['risk_amount']),
                lower_price_range=float(trade_data['lower_price_range']),
                higher_price_range=float(trade_data['higher_price_range']),
                sell_stops=sell_stops,
                order_type=str(trade_data.get('order_type', 'MKT')),
                adaptive_priority=trade_data.get('adaptive_priority'),
                timeout_seconds=int(trade_data.get('timeout_seconds', 30 if str(trade_data.get('order_type', 'MKT')).upper() == 'IBALGO' else 5))
            )
            
            is_valid, error_msg = self._validate_trade(trade)
            if not is_valid:
                return {'success': False, 'error': error_msg}

            # RISK CHECK when adding: ensure we don't enqueue trades that exceed available risk
            if self.available_risk < trade.risk_amount - 1e-9:
                msg = (f"Cannot add trade {trade.ticker}: required risk ${trade.risk_amount:.2f} "
                       f"exceeds available risk ${self.available_risk:.2f}. Update your risk in the Status tab and try again")
                print(f"❌ {msg}")
                return {'success': False, 'error': msg, 'available_risk': self.available_risk}
            
            # Add to database
            if not db_add_trade(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                shares=trade.shares,
                risk_amount=trade.risk_amount,
                lower_price_range=trade.lower_price_range,
                higher_price_range=trade.higher_price_range,
                sell_stops=sell_stops_data,
                order_type=trade.order_type,
                adaptive_priority=trade.adaptive_priority,
                timeout_seconds=trade.timeout_seconds
            ):
                return {'success': False, 'error': 'Failed to add trade to database (may be duplicate)'}
            
            return {
                'success': True,
                'trade_id': trade.trade_id,
                'message': f'Trade added for {trade.ticker}'
            }
            
        except Exception as e:
            self._log_error("ADD_TRADE", trade_data.get('ticker', 'unknown'), str(e), trade_data)
            return {'success': False, 'error': str(e)}
    
    def _get_status_internal(self) -> dict:
        """Internal method to get server status from database"""
        # Get trades from database
        trades_data = db_get_all_trades()
        
        # Calculate server uptime
        uptime_seconds = time.time() - self.server_start_time
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{uptime_hours}h {uptime_minutes}m"
        
        # Format last trade time
        last_trade_formatted = None
        if self.last_trade_time:
            last_trade_formatted = datetime.fromtimestamp(self.last_trade_time).strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            'success': True,
            'available_risk': self.available_risk,
            'active_trades': len(trades_data),
            'server_uptime': uptime_str,
            'last_trade_time': last_trade_formatted,
            'trades': trades_data,
            'is_processing': self.is_processing,
            'error_count': len(self.error_log)
        }

    
    # Public API methods (these queue requests)
    def execute_trade(self, ticker: str, lower_price: float, higher_price: float) -> dict:
        """Execute a specific trade"""
        return self._queue_request('execute_trade', {
            'ticker': ticker,
            'lower_price': lower_price,
            'higher_price': higher_price
        })
    
    def add_trade(self, trade_data: dict) -> dict:
        """Add a new trade"""
        return self._queue_request('add_trade', trade_data)
    
    def get_status(self) -> dict:
        """Get server status"""
        return self._queue_request('get_status')
    
    def get_errors(self) -> List[Dict]:
        """Get error log (direct access, no queuing needed)"""
        return self.error_log.copy()
    
    def update_risk_amount(self, new_amount: float) -> dict:
        """Update available risk amount and persist to database"""
        self.available_risk = new_amount
        save_risk_amount(new_amount)
        return {'success': True, 'available_risk': self.available_risk}
    
    def shutdown(self):
        """Shutdown the server"""
        self.server_running = False
        self.request_queue.put(None)  # Signal shutdown

# Flask app
app = Flask(__name__)
trading_server = StockTradingServer()
flask_cors.CORS(app)  # Enable CORS for all routes

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a specific trade"""
    data = request.json
    
    if not data or 'ticker' not in data or 'lower_price' not in data or 'higher_price' not in data:
        return jsonify({'success': False, 'error': 'Missing required fields: ticker, lower_price, higher_price'}), 400
    
    try:
        ticker = data['ticker']
        lower_price = float(data['lower_price'])
        higher_price = float(data['higher_price'])
        
        if lower_price >= higher_price:
            return jsonify({'success': False, 'error': 'Lower price must be less than higher price'}), 400
        
        result = trading_server.execute_trade(ticker, lower_price, higher_price)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid price values'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/add_trade', methods=['POST'])
def add_trade():
    """Add a new trade"""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    required_fields = ['ticker', 'shares', 'risk_amount', 'lower_price_range', 'higher_price_range', 'sell_stops']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
    
    result = trading_server.add_trade(data)
    
    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify(result), 400
    
@app.route('/remove_trade', methods=['POST'])
def remove_trade():
    """Remove a trade by ID or criteria"""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    trade_id = data.get('trade_id')
    ticker = data.get('ticker')
    lower_price = data.get('lower_price')
    higher_price = data.get('higher_price')
    
    # Must provide either trade_id or all criteria
    if not trade_id and not (ticker and lower_price is not None and higher_price is not None):
        return jsonify({
            'success': False, 
            'error': 'Must provide either trade_id or all criteria (ticker, lower_price, higher_price)'
        }), 400
    
    try:
        if lower_price is not None:
            lower_price = float(lower_price)
        if higher_price is not None:
            higher_price = float(higher_price)
            
        result = trading_server.remove_trade(trade_id, ticker, lower_price, higher_price)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid price values'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get server status"""
    result = trading_server.get_status()
    return jsonify(result), 200

@app.route('/errors', methods=['GET'])
def get_errors():
    """Get error log"""
    errors = trading_server.get_errors()
    return jsonify({'success': True, 'errors': errors}), 200

@app.route('/ib_status', methods=['GET'])
def ib_status():
    """Check IBKR Web API connectivity"""
    sample_symbol = request.args.get('symbol', 'SPY')
    result = trading_server.test_ib_connection(sample_symbol)
    status_code = 200 if result.get('success') else 503
    return jsonify(result), status_code

@app.route('/ib_algos', methods=['GET'])
def ib_algos():
    """List available IBKR algos for a symbol."""
    symbol = request.args.get('symbol', '').strip()
    if not symbol:
        return jsonify({'success': False, 'error': 'Missing required query param: symbol'}), 400

    add_description = request.args.get('addDescription', '1') not in {'0', 'false', 'False', 'no', 'No'}
    add_params = request.args.get('addParams', '1') not in {'0', 'false', 'False', 'no', 'No'}
    algos = request.args.get('algos')

    result = trading_server.get_symbol_algos(symbol, add_description=add_description, add_params=add_params, algos=algos)
    return jsonify(result), 200 if result.get('success') else 503

@app.route('/update_risk', methods=['POST'])
def update_risk():
    """Update available risk amount"""
    data = request.json
    
    if not data or 'amount' not in data:
        return jsonify({'success': False, 'error': 'Missing required field: amount'}), 400
    
    try:
        amount = float(data['amount'])
        result = trading_server.update_risk_amount(amount)
        return jsonify(result), 200
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid amount value'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Stock Trading Server...")
    print("📡 Server will process requests sequentially")
    print("🔗 IBKR connections are managed automatically")
    print("\nAPI Endpoints:")
    print("  POST /execute_trade - Execute a specific trade")
    print("  POST /add_trade - Add a new trade")
    print("  POST /remove_trade - Remove a trade")  # ADD THIS LINE
    print("  GET /status - Get server status")
    print("  GET /errors - Get error log")
    print("  GET /ib_status - Check IBKR Web API connectivity")
    print("  GET /ib_algos?symbol=SPY - List available IBKR algos for symbol")
    print("  POST /update_risk - Update available risk amount")
    print("  GET /health - Health check")

    
    try:
        app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        trading_server.shutdown()
        print("✅ Server shutdown complete")