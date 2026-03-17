from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import CustomColumn, ColumnOrder, CustomColumnValue
from .serializers import (
    CustomColumnSerializer,
    ColumnOrderSerializer,
    CustomColumnValueSerializer,
)


class CustomColumnViewSet(viewsets.ModelViewSet):
    queryset = CustomColumn.objects.all()
    serializer_class = CustomColumnSerializer


class ColumnOrderViewSet(viewsets.ModelViewSet):
    queryset = ColumnOrder.objects.all()
    serializer_class = ColumnOrderSerializer


@api_view(['POST'])
def bulk_update_column_order(request):
    """
    Accepts a list of {column_key, position, is_custom} and upserts them all.
    """
    orders = request.data  # expecting a list
    if not isinstance(orders, list):
        return Response({'error': 'Expected a list of order objects.'}, status=status.HTTP_400_BAD_REQUEST)

    results = []
    for item in orders:
        column_key = item.get('column_key')
        position = item.get('position', 0)
        is_custom = item.get('is_custom', False)
        width = item.get('width', 0)
        is_textarea = item.get('is_textarea', False)
        if not column_key:
            continue
        obj, _ = ColumnOrder.objects.update_or_create(
            column_key=column_key,
            defaults={'position': position, 'is_custom': is_custom, 'width': width, 'is_textarea': is_textarea}
        )
        results.append(ColumnOrderSerializer(obj).data)

    return Response(results, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_custom_column_values(request):
    """
    GET /custom_trade_data/values/?trade_id=1,2,3  or all if no param
    Returns all custom column values, optionally filtered by trade_ids.
    """
    trade_ids_param = request.query_params.get('trade_id', None)
    qs = CustomColumnValue.objects.select_related('column').all()
    if trade_ids_param:
        trade_ids = [int(tid) for tid in trade_ids_param.split(',') if tid.strip().isdigit()]
        qs = qs.filter(trade_id__in=trade_ids)

    serializer = CustomColumnValueSerializer(qs, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def bulk_upsert_custom_column_values(request):
    """
    Accepts a list of {trade_id, column, value} and upserts them.
    """
    values = request.data  # expecting a list
    if not isinstance(values, list):
        return Response({'error': 'Expected a list of value objects.'}, status=status.HTTP_400_BAD_REQUEST)

    results = []
    for item in values:
        trade_id = item.get('trade_id')
        column_id = item.get('column')
        value = item.get('value', '')
        if trade_id is None or column_id is None:
            continue
        obj, _ = CustomColumnValue.objects.update_or_create(
            trade_id=trade_id,
            column_id=column_id,
            defaults={'value': value}
        )
        results.append(CustomColumnValueSerializer(obj).data)

    return Response(results, status=status.HTTP_200_OK)


@api_view(['DELETE'])
def delete_custom_column_values_by_column(request, column_id):
    """Delete all values for a given custom column (used when deleting a column)."""
    deleted_count, _ = CustomColumnValue.objects.filter(column_id=column_id).delete()
    return Response({'deleted': deleted_count}, status=status.HTTP_200_OK)
