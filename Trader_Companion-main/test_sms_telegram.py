"""
Test SMS Alert via Telegram Bot (RECOMMENDED - Works worldwide!)
This is 100% FREE and works for any phone number in any country.
"""

import requests

def send_telegram_alert(bot_token, chat_id, message):
    """
    Send a message via Telegram bot
    
    Args:
        bot_token: Your bot token from @BotFather
        chat_id: Your chat ID (get from @userinfobot)
        message: The alert message to send
    """
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"  # Allows HTML formatting
    }
    
    try:
        print("Sending Telegram message...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("✅ Telegram alert sent successfully!")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TELEGRAM ALERT TEST (RECOMMENDED)")
    print("=" * 60)
    print("\n📱 SETUP STEPS (Takes 2 minutes):")
    print("\n1️⃣  Create a Telegram Bot:")
    print("   - Open Telegram app on your phone")
    print("   - Search for @BotFather")
    print("   - Send: /newbot")
    print("   - Choose a name (e.g., 'My Price Alerts')")
    print("   - Choose a username (e.g., 'my_price_alerts_bot')")
    print("   - Copy the TOKEN you receive (looks like: 123456:ABC-DEF...)")
    print("\n2️⃣  Get Your Chat ID:")
    print("   - Search for @userinfobot in Telegram")
    print("   - Send any message to it")
    print("   - Copy your 'Id' number")
    print("\n3️⃣  Start a chat with your bot:")
    print("   - Search for your bot by its username")
    print("   - Click START or send /start")
    print("\n4️⃣  Edit this file and add your credentials below:\n")
    
    # ⚠️ ADD YOUR CREDENTIALS HERE:
    BOT_TOKEN = "8575601323:AAFLZzgLCO2LMQx6Kj7UKZHPybmuR-_s2XA"  # From @BotFather
    CHAT_ID = "8586730851"  # From @userinfobot
    
    # Check if credentials are set
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("⚠️  Please edit this file and add your BOT_TOKEN and CHAT_ID first!")
        print("\nAfter adding them, run: python test_sms_telegram.py")
    else:
        # Test message with HTML formatting
        test_message = """
🚨 <b>PRICE ALERT TEST</b>

Symbol: AAPL
Type: Above threshold
Price: $150.00
Time: 16:39:40

This is a test alert from your Price Alerts app!
        """
        
        send_telegram_alert(BOT_TOKEN, CHAT_ID, test_message)
        print("\n✅ Check your Telegram app for the message!")
