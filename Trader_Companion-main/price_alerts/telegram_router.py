"""
Database router for Telegram configuration.
Routes TelegramConfig model to telegram_config_db.
"""


class TelegramConfigRouter:
    """Router to direct TelegramConfig model to telegram_config_db"""
    
    route_app_labels = {'telegram_config'}
    
    def db_for_read(self, model, **hints):
        """Direct read operations for TelegramConfig to telegram_config_db"""
        if model._meta.app_label == 'price_alerts' and model.__name__ == 'TelegramConfig':
            return 'telegram_config_db'
        return None
    
    def db_for_write(self, model, **hints):
        """Direct write operations for TelegramConfig to telegram_config_db"""
        if model._meta.app_label == 'price_alerts' and model.__name__ == 'TelegramConfig':
            return 'telegram_config_db'
        return None
    
    def allow_relation(self, obj1, obj2, **hints):
        """Allow relations within telegram_config_db"""
        if (obj1._meta.app_label == 'price_alerts' and obj1.__class__.__name__ == 'TelegramConfig') or \
           (obj2._meta.app_label == 'price_alerts' and obj2.__class__.__name__ == 'TelegramConfig'):
            return True
        return None
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """Ensure TelegramConfig migrations only apply to telegram_config_db"""
        if model_name == 'telegramconfig':
            return db == 'telegram_config_db'
        return None
