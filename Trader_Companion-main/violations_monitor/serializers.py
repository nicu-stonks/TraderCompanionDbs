from rest_framework import serializers
from .models import MonitoredTrade, HistoricalPrice, MonitorPreferences


class MonitoredTradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitoredTrade
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class MonAlertTradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitoredTrade
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class HistoricalPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = HistoricalPrice
        fields = '__all__'


class MonitorPreferencesSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitorPreferences
        fields = '__all__'
        read_only_fields = ['id', 'updated_at']


class MonAlertPreferencesSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitorPreferences
        fields = '__all__'
        read_only_fields = ['id', 'updated_at']
