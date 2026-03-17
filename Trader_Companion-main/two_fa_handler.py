from ibeam.src.two_fa_handlers.two_fa_handler import TwoFaHandler
import pyotp
import logging

# 1. PASTE YOUR SAVED SEED HERE (Remove spaces if any)
SECURE_SEED = "HK3VM6ZUNXBJ7DEJKSY55L37ERDD6GCS" 

class CustomHandler(TwoFaHandler):
    def get_two_fa_code(self, driver) -> str:
        # This function is called by IBeam when it needs a code
        logging.info("🤖 CUSTOM HANDLER: Generating 2FA code...")
        
        try:
            totp = pyotp.TOTP(SECURE_SEED)
            code = totp.now()
            logging.info(f"🤖 CUSTOM HANDLER: Code generated: {code}")
            return code
        except Exception as e:
            logging.error(f"🤖 CUSTOM HANDLER Error: {e}")
            return None