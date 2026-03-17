from rest_framework import serializers
from .models import LLMConfig, Supplement, DailyRecord, FoodItem

class LLMConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLMConfig
        fields = ['model_name', 'api_key', 'user_profile', 'available_models']

class SupplementSerializer(serializers.ModelSerializer):
    class Meta:
        model = Supplement
        fields = ['id', 'name', 'details']

class FoodItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = FoodItem
        fields = ['id', 'name', 'details']

class DailyRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = DailyRecord
        fields = ['date', 'foods_eaten', 'supplements_taken', 'nutrient_completion', 'nutrient_sources', 'recommendations', 'chat_history']
