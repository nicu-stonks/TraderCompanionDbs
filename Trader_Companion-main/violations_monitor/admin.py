from django.contrib import admin
from .models import MonitoredTrade, HistoricalPrice, MonitorPreferences

admin.site.register(MonitoredTrade)
admin.site.register(HistoricalPrice)
admin.site.register(MonitorPreferences)
