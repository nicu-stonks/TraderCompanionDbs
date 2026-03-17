from django.db import models


class CustomColumn(models.Model):
    """User-defined custom text column for the trades table."""
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return self.name


class ColumnOrder(models.Model):
    """Stores the display order of all columns (both default and custom)."""
    id = models.AutoField(primary_key=True)
    column_key = models.CharField(max_length=100, unique=True)  # e.g. 'Ticker', 'custom_3'
    position = models.IntegerField(default=0)
    is_custom = models.BooleanField(default=False)
    width = models.IntegerField(default=0)  # pixel width, 0 means use default
    is_textarea = models.BooleanField(default=False)  # render as expandable textarea

    class Meta:
        ordering = ['position']

    def __str__(self):
        return f"{self.column_key} @ position {self.position}"


class CustomColumnValue(models.Model):
    """Stores the value of a custom column for a specific trade."""
    id = models.AutoField(primary_key=True)
    trade_id = models.IntegerField()  # References trades_history.Trades.ID
    column = models.ForeignKey(CustomColumn, on_delete=models.CASCADE, related_name='values')
    value = models.TextField(blank=True, default='')

    class Meta:
        unique_together = ('trade_id', 'column')
        ordering = ['trade_id', 'column']

    def __str__(self):
        return f"Trade {self.trade_id} - {self.column.name}: {self.value[:50]}"
