"""
Telegram notification service for price alerts.
Sends alert messages via Telegram Bot API.
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def send_telegram_alert(bot_token, chat_id, ticker, alert_price, current_price, percent_change=None, alert_type="threshold"):
    """
    Send a price alert notification via Telegram.
    
    Args:
        bot_token: Bot token from @BotFather
        chat_id: User's chat ID from @userinfobot
        ticker: Stock ticker symbol
        alert_price: The alert threshold price
        current_price: Current stock price
        percent_change: Percent change from previous close (optional)
        alert_type: Type of alert (e.g., "threshold", "above", "below")
    
    Returns:
        bool: True if message sent successfully, False otherwise
    """
    try:
        # Lazy import to avoid failing if requests not installed
        import requests
    except ImportError:
        logger.error("requests library not installed - cannot send Telegram notifications")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Format message with HTML, including percent change if available
    percent_change_text = ""
    if percent_change is not None:
        sign = "+" if percent_change >= 0 else ""
        percent_change_text = f"\n<b>Change:</b> {sign}{percent_change:.2f}%"
    
    message = f"""
🔔 <b>PRICE ALERT TRIGGERED</b> 🔔

<b>Symbol:</b> {ticker}
<b>Alert Price:</b> ${alert_price:.2f}
<b>Current Price:</b> ${current_price:.2f}{percent_change_text}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
    """.strip()
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Telegram alert sent successfully for {ticker}")
            return True
        else:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Telegram API request timed out")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram API request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram alert: {e}")
        return False


def test_telegram_connection(bot_token, chat_id):
    """
    Test Telegram bot connection by sending a test message.
    
    Args:
        bot_token: Bot token from @BotFather
        chat_id: User's chat ID from @userinfobot
    
    Returns:
        dict: {"success": bool, "message": str}
    """
    try:
        # Lazy import to avoid failing if requests not installed
        import requests
    except ImportError:
        return {
            "success": False,
            "message": "requests library not installed. Please install it with: pip install requests"
        }
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    message = """
✅ <b>Connection Test Successful!</b>

Your Telegram bot is configured correctly and ready to send price alerts.

You can now enable notifications in the settings.
    """.strip()
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": "Test message sent successfully! Check your Telegram app."
            }
        elif response.status_code == 401:
            return {
                "success": False,
                "message": "Invalid bot token. Please check your token from @BotFather."
            }
        elif response.status_code == 400:
            response_data = response.json()
            if "chat not found" in response_data.get("description", "").lower():
                return {
                    "success": False,
                    "message": "Chat ID not found. Make sure you've started a conversation with your bot first."
                }
            return {
                "success": False,
                "message": f"Invalid request: {response_data.get('description', 'Unknown error')}"
            }
        else:
            return {
                "success": False,
                "message": f"Telegram API error: {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "Connection timeout. Please check your internet connection."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }
