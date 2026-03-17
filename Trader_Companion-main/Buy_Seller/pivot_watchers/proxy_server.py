#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import logging
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes to allow frontend connections

class TradingBotManager:
    def __init__(self, script_path: str = "price_going_up_optional_volume_script.py"):
        self.script_path = script_path
        
    def validate_script_exists(self) -> bool:
        """Check if the trading bot script exists"""
        return os.path.exists(self.script_path)
    
    def build_command(self, params: Dict) -> List[str]:
        """Build the command to execute the trading bot"""
        if not self.validate_script_exists():
            raise FileNotFoundError(f"Trading bot script not found at: {self.script_path}")
        
        # Required parameters
        ticker = params.get('ticker', '').upper()
        lower_price = params.get('lower_price')
        higher_price = params.get('higher_price')
        
        if not ticker or lower_price is None or higher_price is None:
            raise ValueError("Missing required parameters: ticker, lower_price, higher_price")
        
        # Build base command
        cmd = [
            "powershell.exe",
            "-Command",
            f"python {self.script_path} {ticker} {lower_price} {higher_price}"
        ]
        
        # Add optional parameters
        volume_requirements = params.get('volume_requirements', [])
        for vol_req in volume_requirements:
            cmd[2] += f" --volume {vol_req}"
        
        if params.get('pivot_adjustment'):
            cmd[2] += f" --pivot-adjustment {params['pivot_adjustment']}"
        
    # Removed legacy average momentum parameters
        if params.get('breakout_lookback_minutes') is not None:
            cmd[2] += f" --breakout-lookback-minutes {params['breakout_lookback_minutes']}"
        if params.get('breakout_exclude_minutes') is not None:
            cmd[2] += f" --breakout-exclude-minutes {params['breakout_exclude_minutes']}"
        
        if params.get('day_high_max_percent_off'):
            cmd[2] += f" --day-high-max-percent-off {params['day_high_max_percent_off']}"

        if params.get('max_day_low'):
            cmd[2] += f" --max-day-low {params['max_day_low']}"
            
        if params.get('min_day_low'):
            cmd[2] += f" --min-day-low {params['min_day_low']}"



        if params.get('time_in_pivot'):
            cmd[2] += f" --time-in-pivot {params['time_in_pivot']}"
        
        if params.get('time_in_pivot_positions'):
            cmd[2] += f" --time-in-pivot-positions {params['time_in_pivot_positions']}"
        
        if params.get('volume_multipliers') and len(params['volume_multipliers']) == 3:
            multipliers_str = ' '.join(str(m) for m in params['volume_multipliers'])
            cmd[2] += f" --volume-multipliers {multipliers_str}"
        
        if params.get('data_server'):
            cmd[2] += f" --data-server {params['data_server']}"
        
        if params.get('trade_server'):
            cmd[2] += f" --trade-server {params['trade_server']}"

    # Removed momentum_required_at_open handling

        # Additional custom wait after market open (float minutes)
        if params.get('wait_after_open_minutes'):
            cmd[2] += f" --wait-after-open {params['wait_after_open_minutes']}"

        # Late-day trading window controls (NEW 2025-09-05)
        if params.get('start_minutes_before_close') is not None:
            cmd[2] += f" --start-minutes-before-close {params['start_minutes_before_close']}"
        if params.get('stop_minutes_before_close') is not None:
            cmd[2] += f" --stop-minutes-before-close {params['stop_minutes_before_close']}"

        # NEW (2025-09-07): Optional request trade price overrides
        if params.get('request_lower_price') is not None:
            cmd[2] += f" --request-lower-price {params['request_lower_price']}"
        if params.get('request_higher_price') is not None:
            cmd[2] += f" --request-higher-price {params['request_higher_price']}"
        
        return cmd
    
    def start_bot(self, params: Dict) -> Dict:
        """Start a new trading bot instance without tracking it"""
        try:
            # Build command
            cmd = self.build_command(params)
            
            # Start the process in a new PowerShell window
            # Use CREATE_NEW_CONSOLE to open in new terminal window
            # Don't redirect stdout/stderr so output appears in the terminal
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            logger.info(f"Started bot PID {process.pid} for {params.get('ticker', 'UNKNOWN')}")

            return {
                'success': True,
                'pid': process.pid,
                'message': f"Trading bot started for {params.get('ticker', 'UNKNOWN')}",
                'started_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize the bot manager
bot_manager = TradingBotManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'script_exists': bot_manager.validate_script_exists(),
        'script_path': bot_manager.script_path
    })

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start a new trading bot instance"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        result = bot_manager.start_bot(data)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in start_bot endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    

if __name__ == '__main__':
    # Check if script exists on startup
    if not bot_manager.validate_script_exists():
        logger.warning(f"Trading bot script not found at: {bot_manager.script_path}")
        logger.warning("Please ensure the script is in the same directory or update the path")
    
    logger.info("Starting Trading Bot Proxy Server...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /start_bot - Start a new bot (no tracking)")
    
    app.run(host='0.0.0.0', port=5003, debug=True)