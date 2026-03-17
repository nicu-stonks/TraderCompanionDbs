"""
Test SMS Alert via Email-to-SMS Gateway
NOTE: This primarily works for US carriers. For Romanian carriers, use the Telegram version instead.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_sms_via_email(phone_number, carrier_gateway, message):
    """
    Send SMS via email-to-SMS gateway
    
    Common US carrier gateways:
    - AT&T: txt.att.net
    - Verizon: vtext.com
    - T-Mobile: tmomail.net
    - Sprint: messaging.sprintpcs.com
    """
    
    # Gmail credentials - YOU NEED TO SET THESE
    gmail_user = "your-email@gmail.com"  # ⚠️ CHANGE THIS
    gmail_app_password = "your-16-char-app-password"  # ⚠️ CHANGE THIS (NOT your regular password!)
    
    # Construct the SMS email address
    to_address = f"{phone_number}@{carrier_gateway}"
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = to_address
        msg['Subject'] = "Price Alert Test"
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Connect to Gmail's SMTP server
        print("Connecting to Gmail SMTP server...")
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        
        # Login
        print("Logging in...")
        server.login(gmail_user, gmail_app_password)
        
        # Send email
        print(f"Sending SMS to {to_address}...")
        server.send_message(msg)
        
        # Close connection
        server.quit()
        
        print("✅ SMS sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error sending SMS: {e}")
        return False


if __name__ == "__main__":
    # Your phone number
    phone = "0735535918"
    
    # Example for different carriers (uncomment the one you need):
    # carrier = "txt.att.net"  # AT&T
    # carrier = "vtext.com"  # Verizon
    # carrier = "tmomail.net"  # T-Mobile
    carrier = "txt.att.net"  # ⚠️ CHANGE THIS to your carrier
    
    # Test message
    test_message = "🚨 PRICE ALERT TEST: AAPL triggered at $150.00"
    
    print("=" * 50)
    print("SMS ALERT TEST")
    print("=" * 50)
    print("\n⚠️  IMPORTANT SETUP STEPS:")
    print("1. Edit this file and add your Gmail credentials")
    print("2. Enable 2FA on your Google account")
    print("3. Create an App Password at: https://myaccount.google.com/apppasswords")
    print("4. Use the 16-character app password (spaces removed)")
    print("5. Set the correct carrier gateway for your phone\n")
    
    send_sms_via_email(phone, carrier, test_message)
