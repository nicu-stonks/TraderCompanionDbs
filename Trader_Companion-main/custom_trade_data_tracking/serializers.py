from rest_framework import serializers
from .models import CustomColumn, ColumnOrder, CustomColumnValue


class CustomColumnSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomColumn
        fields = '__all__'


class ColumnOrderSerializer(serializers.ModelSerializer):
    class Meta:
        model = ColumnOrder
        fields = '__all__'


class CustomColumnValueSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomColumnValue
        fields = '__all__'


class BulkColumnOrderSerializer(serializers.Serializer):
    """Accepts a list of {column_key, position, is_custom} objects."""
    orders = ColumnOrderSerializer(many=True)


class BulkCustomColumnValueSerializer(serializers.Serializer):
    """Accepts a list of {trade_id, column_id, value} for bulk upsert."""
    values = CustomColumnValueSerializer(many=True)
