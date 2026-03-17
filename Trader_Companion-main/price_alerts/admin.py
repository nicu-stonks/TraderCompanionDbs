from django.contrib import admin
from .models import Alert, AlarmSettings


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['ticker', 'alert_price', 'current_price', 'is_active', 'triggered', 'created_at']
    list_filter = ['is_active', 'triggered', 'created_at']
    search_fields = ['ticker']
    readonly_fields = ['triggered', 'triggered_at', 'current_price', 'last_checked', 'initial_price_above_alert']


@admin.register(AlarmSettings)
class AlarmSettingsAdmin(admin.ModelAdmin):
    list_display = ['alarm_sound_path', 'play_duration', 'pause_duration', 'cycles']
    
    def has_add_permission(self, request):
        # Only allow one settings object
        return not AlarmSettings.objects.exists()
    
    def has_delete_permission(self, request, obj=None):
        # Don't allow deletion of settings
        return False
