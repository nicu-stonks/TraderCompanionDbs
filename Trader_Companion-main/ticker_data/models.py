from django.db import models
from django.utils import timezone

class ProviderSettings(models.Model):
    """
    Singleton model for controlling global fetcher settings.
    """
    max_requests_per_10s = models.FloatField(default=10.0, help_text="Max permitted requests per 10 seconds.")
    active_provider = models.CharField(max_length=20, default='yfinance', choices=[('yfinance', 'YFinance'), ('webull', 'Webull')], help_text="The currently active data provider. Can auto-fallback.")
    updated_at = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        # Enforce singleton pattern
        self.pk = 1
        super().save(*args, **kwargs)
        
    @classmethod
    def load(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return f"Provider Settings (Max Requests per 10s: {self.max_requests_per_10s}, Provider: {self.active_provider})"


class TrackedTicker(models.Model):
    """
    Tickers that the orchestrator should continuously monitor and fetch updates for in the background loop.
    """
    symbol = models.CharField(max_length=20, primary_key=True)
    added_at = models.DateTimeField(auto_now_add=True)
    last_trade_seen_at = models.DateTimeField(null=True, blank=True, help_text="Used to prune inactive tickers.")

    def __str__(self):
        return self.symbol


class HistoricalPrice(models.Model):
    """
    Stores the daily historical bars (EOD data).
    """
    symbol = models.CharField(max_length=20, db_index=True)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    ingested_at = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        unique_together = ('symbol', 'date')
        ordering = ['-date']
        
    def __str__(self):
        return f"{self.symbol} - {self.date}"


class HistoricalPrice5m(models.Model):
    """
    Stores the 5-minute historical bars (Intraday data).
    """
    symbol = models.CharField(max_length=20, db_index=True)
    timestamp = models.DateTimeField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    ingested_at = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        unique_together = ('symbol', 'timestamp')
        ordering = ['-timestamp']
        
    def __str__(self):
        return f"{self.symbol} - {self.timestamp}"


class HistoricalPriceWeekly(models.Model):
    """
    Stores weekly historical bars (used for SPY market timing chart timeframe toggle).
    """
    symbol = models.CharField(max_length=20, db_index=True)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    ingested_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        unique_together = ('symbol', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.symbol} - {self.date} (1w)"


class RequestLog(models.Model):
    """
    One row per ticker fetch event.
    request_count stores how many underlying API calls that fetch consumed
    so requests = SUM(request_count) and tickers = COUNT(rows).
    """
    provider = models.CharField(max_length=20, db_index=True)
    timestamp = models.FloatField(db_index=True, help_text="time.time() float timestamp of the request.")
    symbol = models.CharField(max_length=20, null=True, blank=True)
    duration_ms = models.FloatField(null=True, blank=True)
    success = models.BooleanField(default=True)
    request_count = models.IntegerField(default=1)
    
    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.provider}] {self.symbol} at {self.timestamp}"
