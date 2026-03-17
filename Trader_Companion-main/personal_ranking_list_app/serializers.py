import json
from rest_framework import serializers
from .models import (
    RankingBox,
    StockPick,
    UserPageState,
    GlobalCharacteristic,
    StockPickCharacteristic,
    OrderedCharacteristic,
    PriorityCharacteristic,
    ColorCodedCharacteristic
)


class GlobalCharacteristicSerializer(serializers.ModelSerializer):
    default_score = serializers.FloatField()

    class Meta:
        model = GlobalCharacteristic
        fields = ['id', 'name', 'default_score', 'created_at']


class OrderedCharacteristicSerializer(serializers.ModelSerializer):
    characteristic_id = serializers.PrimaryKeyRelatedField(
        source='characteristic', queryset=GlobalCharacteristic.objects.all()
    )
    name = serializers.ReadOnlyField(source='characteristic.name')

    class Meta:
        model = OrderedCharacteristic
        fields = ['id', 'characteristic_id', 'name', 'position']


class PriorityCharacteristicSerializer(serializers.ModelSerializer):
    characteristic_id = serializers.PrimaryKeyRelatedField(
        source='characteristic', queryset=GlobalCharacteristic.objects.all()
    )
    name = serializers.ReadOnlyField(source='characteristic.name')

    class Meta:
        model = PriorityCharacteristic
        fields = ['id', 'characteristic_id', 'name', 'created_at']


class ColorCodedCharacteristicSerializer(serializers.ModelSerializer):
    characteristic_id = serializers.PrimaryKeyRelatedField(
        source='characteristic', queryset=GlobalCharacteristic.objects.all()
    )
    name = serializers.ReadOnlyField(source='characteristic.name')

    class Meta:
        model = ColorCodedCharacteristic
        fields = ['id', 'characteristic_id', 'name', 'created_at']


class StockPickCharacteristicSerializer(serializers.ModelSerializer):
    name = serializers.ReadOnlyField(source='characteristic.name')
    characteristic_id = serializers.PrimaryKeyRelatedField(
        source='characteristic',
        queryset=GlobalCharacteristic.objects.all()
    )
    score = serializers.FloatField()

    class Meta:
        model = StockPickCharacteristic
        fields = ['id', 'characteristic_id', 'name', 'score']


class StockPickSerializer(serializers.ModelSerializer):
    characteristics = StockPickCharacteristicSerializer(
        source='stock_characteristics',
        many=True,
        read_only=True
    )
    ranking_box = serializers.PrimaryKeyRelatedField(queryset=RankingBox.objects.all())
    total_score = serializers.FloatField()
    personal_opinion_score = serializers.FloatField(required=False, default=0)
    demand_reason = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    case_text = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    note = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    class Meta:
        model = StockPick
        fields = ['id', 'ranking_box', 'symbol', 'total_score', 'personal_opinion_score',
                  'demand_reason', 'case_text', 'note', 'created_at', 'characteristics']  # Include note in fields

class RankingBoxSerializer(serializers.ModelSerializer):
    stock_picks = StockPickSerializer(many=True, read_only=True)

    class Meta:
        model = RankingBox
        fields = ['id', 'title', 'created_at', 'stock_picks']


class UserPageStateSerializer(serializers.ModelSerializer):
    ranking_boxes_order = serializers.JSONField(required=False)

    class Meta:
        model = UserPageState
        fields = ['id', 'column_count', 'ranking_boxes_order', 'updated_at']

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        try:
            if isinstance(ret['ranking_boxes_order'], str):
                ret['ranking_boxes_order'] = json.loads(ret['ranking_boxes_order'])
            if not isinstance(ret['ranking_boxes_order'], list):
                ret['ranking_boxes_order'] = []
        except (json.JSONDecodeError, TypeError):
            ret['ranking_boxes_order'] = []
        return ret

    def to_internal_value(self, data):
        if 'ranking_boxes_order' in data:
            if isinstance(data['ranking_boxes_order'], list):
                data = data.copy()
                data['ranking_boxes_order'] = json.dumps(data['ranking_boxes_order'])
            elif isinstance(data['ranking_boxes_order'], str):
                try:
                    json.loads(data['ranking_boxes_order'])
                except json.JSONDecodeError:
                    raise serializers.ValidationError({
                        'ranking_boxes_order': ['Invalid JSON format']
                    })
        return super().to_internal_value(data)