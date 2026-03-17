from django.db import models
from django.core.validators import MinValueValidator


class Alert(models.Model):
    """Model for price alerts"""
    ticker = models.CharField(max_length=10)
    alert_price = models.FloatField(validators=[MinValueValidator(0)])
    is_active = models.BooleanField(default=True)
    triggered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    triggered_at = models.DateTimeField(null=True, blank=True)
    current_price = models.FloatField(null=True, blank=True)
    last_checked = models.DateTimeField(null=True, blank=True)
    
    # Track the initial price relationship to determine trigger direction
    initial_price_above_alert = models.BooleanField(null=True, blank=True)
    
    # Store previous day's close and percent change
    previous_close = models.FloatField(null=True, blank=True)
    percent_change = models.FloatField(null=True, blank=True)
    
    # User-configurable fetch interval in milliseconds (minimum 500ms, default 1000ms)
    fetch_interval_ms = models.IntegerField(default=1000, validators=[MinValueValidator(500)])
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.ticker} @ ${self.alert_price:.2f}"


class AlarmSettings(models.Model):
    """Model for user alarm preferences"""
    # Alarm sound file path (relative to alarm_sounds folder or absolute)
    alarm_sound_path = models.CharField(max_length=500, default='alarm-clock-2.mp3')
    
    # Play duration in seconds
    play_duration = models.IntegerField(default=5, validators=[MinValueValidator(1)])
    
    # Pause duration in seconds between cycles
    pause_duration = models.IntegerField(default=3, validators=[MinValueValidator(0)])
    
    # Number of cycles (how many times to repeat play-pause)
    cycles = models.IntegerField(default=3, validators=[MinValueValidator(1)])
    
    # Ensure only one settings object exists
    class Meta:
        verbose_name_plural = "Alarm Settings"
    
    def __str__(self):
        return f"Alarm: {self.alarm_sound_path} ({self.cycles} cycles)"
    
    @classmethod
    def get_settings(cls):
        """Get or create the single settings instance"""
        obj, created = cls.objects.get_or_create(id=1)
        return obj


class TelegramConfig(models.Model):
    """Model for Telegram notification configuration"""
    # Bot token from @BotFather
    bot_token = models.CharField(max_length=500, blank=True, default='')
    
    # Chat ID from @userinfobot
    chat_id = models.CharField(max_length=100, blank=True, default='')
    
    # Enable/disable Telegram notifications
    enabled = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "Telegram Configuration"
    
    def __str__(self):
        status = "Enabled" if self.enabled else "Disabled"
        configured = "Configured" if self.bot_token and self.chat_id else "Not Configured"
        return f"Telegram Notifications: {status} ({configured})"
    
    @classmethod
    def get_config(cls):
        """Get or create the single config instance"""
        obj, created = cls.objects.using('telegram_config_db').get_or_create(id=1)
        return obj
    
    def save(self, *args, **kwargs):
        """Override save to use telegram_config_db"""
        kwargs['using'] = 'telegram_config_db'
        super().save(*args, **kwargs)
