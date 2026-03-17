from rest_framework import serializers
from .models import Alert, AlarmSettings


class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = '__all__'
        read_only_fields = ['triggered', 'triggered_at', 'current_price', 'last_checked', 'initial_price_above_alert', 'previous_close', 'percent_change']


class AlarmSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlarmSettings
        fields = '__all__'

