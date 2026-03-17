from django.db import models


class MonitoredTrade(models.Model):
    """A stock position being monitored for violations/confirmations."""
    ticker = models.CharField(max_length=10)
    start_date = models.DateField(help_text="Buy date / start of monitoring")
    end_date = models.DateField(null=True, blank=True, help_text="Custom end date")
    use_latest_end_date = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.ticker} ({self.start_date})"


class HistoricalPrice(models.Model):
    """Cached daily OHLCV data for a ticker."""
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('ticker', 'date')
        ordering = ['date']
        indexes = [
            models.Index(fields=['ticker', 'date']),
        ]

    def __str__(self):
        return f"{self.ticker} {self.date}: C={self.close}"


class MonitorPreferences(models.Model):
    """Singleton storing which violation/confirmation checks are enabled."""
    preferences = models.JSONField(default=dict, help_text="Map of check_key -> enabled boolean")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Monitor Preferences"

    def __str__(self):
        return f"Preferences (updated {self.updated_at})"


# MonAlert aliases (non-breaking): keep DB models intact while exposing clearer naming.
MonAlertTrade = MonitoredTrade
MonAlertPreferences = MonitorPreferences
