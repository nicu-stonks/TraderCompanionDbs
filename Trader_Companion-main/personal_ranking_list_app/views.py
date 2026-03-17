from django.db import models, transaction
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from decimal import Decimal


from .models import (
    UserPageState,
    StockPick,
    RankingBox,
    GlobalCharacteristic,
    StockPickCharacteristic,
    OrderedCharacteristic,
    PriorityCharacteristic,
    ColorCodedCharacteristic
)
from .serializers import (
    UserPageStateSerializer,
    StockPickSerializer,
    RankingBoxSerializer,
    GlobalCharacteristicSerializer,
    OrderedCharacteristicSerializer,
    PriorityCharacteristicSerializer,
    ColorCodedCharacteristicSerializer
)


class RankingBoxViewSet(viewsets.ModelViewSet):
    queryset = RankingBox.objects.all()
    serializer_class = RankingBoxSerializer

    @action(detail=True, methods=['get'])
    def stock_picks(self, request, pk=None):
        ranking_box = self.get_object()
        stock_picks = ranking_box.stock_picks.all()
        serializer = StockPickSerializer(stock_picks, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['delete'])
    def delete_all_stocks(self, request, pk=None):
        """Delete all stock picks in this box"""
        ranking_box = self.get_object()
        ranking_box.stock_picks.all().delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        """Delete all ranking boxes"""
        RankingBox.objects.all().delete()
        # Also reset user page state order
        UserPageState.objects.update(ranking_boxes_order='[]')
        return Response(status=status.HTTP_204_NO_CONTENT)


class GlobalCharacteristicViewSet(viewsets.ModelViewSet):
    queryset = GlobalCharacteristic.objects.all()
    serializer_class = GlobalCharacteristicSerializer

    @action(detail=True, methods=['get'])
    def stock_picks(self, request, pk=None):
        """Get all stock picks that have this characteristic"""
        characteristic = self.get_object()
        stock_picks = characteristic.stock_picks.all()
        serializer = StockPickSerializer(stock_picks, many=True)
        return Response(serializer.data)


class StockPickViewSet(viewsets.ModelViewSet):
    queryset = StockPick.objects.all()
    serializer_class = StockPickSerializer

    def get_queryset(self):
        queryset = StockPick.objects.all().prefetch_related('stock_characteristics__characteristic')
        ranking_box_id = self.request.query_params.get('ranking_box', None)
        if ranking_box_id is not None:
            queryset = queryset.filter(ranking_box_id=ranking_box_id)
        return queryset

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        """Delete all stock picks across all boxes"""
        StockPick.objects.all().delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    def update_total_score(self, stock_pick):
        """Helper method to update the total score of a stock pick using ORM"""
        from django.db.models import Sum

        # Calculate the sum of characteristic scores using ORM
        characteristics_score = stock_pick.stock_characteristics.aggregate(Sum('score'))['score__sum'] or 0

        # Convert to Decimal if needed (or make both values float)
        if isinstance(characteristics_score, Decimal) and not isinstance(stock_pick.personal_opinion_score, Decimal):
            personal_score = Decimal(str(stock_pick.personal_opinion_score))
        else:
            personal_score = stock_pick.personal_opinion_score

        # Add the personal opinion score to the total
        total_score = characteristics_score + personal_score

        # Update the stock_pick total_score
        stock_pick.total_score = total_score
        stock_pick.save(update_fields=['total_score'])
        return stock_pick

    @action(detail=True, methods=['post'])
    def update_personal_score(self, request, pk=None):
        """Update the personal opinion score for a stock pick"""
        stock_pick = self.get_object()

        if 'personal_opinion_score' not in request.data:
            return Response(
                {"personal_opinion_score": ["This field is required."]},
                status=status.HTTP_400_BAD_REQUEST
            )

        personal_score = request.data['personal_opinion_score']

        try:
            # Convert to float to ensure it's a valid number
            personal_score = float(personal_score)
        except (ValueError, TypeError):
            return Response(
                {"personal_opinion_score": ["Must be a valid number."]},
                status=status.HTTP_400_BAD_REQUEST
            )

        with transaction.atomic():
            # Update the personal opinion score
            stock_pick.personal_opinion_score = personal_score
            stock_pick.save(update_fields=['personal_opinion_score'])

            # Update the total score
            self.update_total_score(stock_pick)

        # Refresh the object to get the latest data
        stock_pick.refresh_from_db()

        # Return the updated stock pick
        serializer = self.get_serializer(stock_pick)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def add_characteristic(self, request, pk=None):
        """Add a characteristic to a stock pick"""
        stock_pick = self.get_object()

        if 'characteristic_id' not in request.data:
            return Response(
                {"characteristic_id": ["This field is required."]},
                status=status.HTTP_400_BAD_REQUEST
            )

        characteristic_id = request.data['characteristic_id']
        characteristic = get_object_or_404(GlobalCharacteristic, pk=characteristic_id)

        # Get score from request or use default
        score = request.data.get('score', characteristic.default_score)

        with transaction.atomic():
            # Check if characteristic already exists for this stock
            existing = stock_pick.stock_characteristics.filter(characteristic=characteristic).first()

            if existing:
                # Update existing
                existing.score = score
                existing.save(update_fields=['score'])
            else:
                # Create new
                StockPickCharacteristic.objects.create(
                    stockpick=stock_pick,
                    characteristic=characteristic,
                    score=score
                )

            # Update total score
            self.update_total_score(stock_pick)

        # Refresh the object to get the latest data
        stock_pick.refresh_from_db()

        # Return the updated stock pick
        serializer = self.get_serializer(stock_pick)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def remove_characteristic(self, request, pk=None):
        """Remove a characteristic from a stock pick"""
        stock_pick = self.get_object()

        if 'characteristic_id' not in request.data:
            return Response(
                {"characteristic_id": ["This field is required."]},
                status=status.HTTP_400_BAD_REQUEST
            )

        characteristic_id = request.data['characteristic_id']

        with transaction.atomic():
            # Delete the characteristic
            deleted_count, _ = stock_pick.stock_characteristics.filter(characteristic_id=characteristic_id).delete()

            if deleted_count == 0:
                return Response(
                    {"detail": "Characteristic not found for this stock pick"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Update total score
            self.update_total_score(stock_pick)

        # Refresh the object to get the latest data
        stock_pick.refresh_from_db()

        # Return the updated stock pick
        serializer = self.get_serializer(stock_pick)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def set_characteristics(self, request, pk=None):
        """Set characteristics for a stock pick"""
        stock_pick = self.get_object()
        characteristics_data = request.data.get('characteristics', [])

        with transaction.atomic():
            # Delete existing characteristics
            stock_pick.stock_characteristics.all().delete()

            # Create new characteristics
            for char_data in characteristics_data:
                characteristic = get_object_or_404(GlobalCharacteristic, pk=char_data['characteristic_id'])
                score = char_data.get('score', characteristic.default_score)

                StockPickCharacteristic.objects.create(
                    stockpick=stock_pick,
                    characteristic=characteristic,
                    score=score
                )

            # Update total score
            self.update_total_score(stock_pick)

        # Refresh the object to get the latest data
        stock_pick.refresh_from_db()

        # Return updated stock pick
        serializer = self.get_serializer(stock_pick)
        return Response(serializer.data)


class UserPageStateViewSet(viewsets.ModelViewSet):
    queryset = UserPageState.objects.all()
    serializer_class = UserPageStateSerializer

    def get_object(self):
        obj, created = UserPageState.objects.get_or_create(
            pk=1,
            defaults={
                'column_count': 3,
                'ranking_boxes_order': '[]'
            }
        )
        return obj

    def list(self, request):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    def create(self, request):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class OrderedCharacteristicViewSet(viewsets.ModelViewSet):
    queryset = OrderedCharacteristic.objects.all().select_related('characteristic')
    serializer_class = OrderedCharacteristicSerializer

    def create(self, request, *args, **kwargs):
        # Auto-assign next position if not provided
        data = request.data.copy()
        if 'position' not in data or data['position'] in (None, ''):
            last = OrderedCharacteristic.objects.order_by('-position').first()
            data['position'] = (last.position + 1) if last else 1
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'])
    def reorder(self, request):
        """Bulk update ordering: expects list of {id, position}."""
        items = request.data.get('items', [])
        with transaction.atomic():
            for item in items:
                OrderedCharacteristic.objects.filter(pk=item.get('id')).update(position=item.get('position'))
        return Response({'status': 'ok'})


class PriorityCharacteristicViewSet(viewsets.ModelViewSet):
    queryset = PriorityCharacteristic.objects.all().select_related('characteristic')
    serializer_class = PriorityCharacteristicSerializer


class ColorCodedCharacteristicViewSet(viewsets.ModelViewSet):
    queryset = ColorCodedCharacteristic.objects.all().select_related('characteristic')
    serializer_class = ColorCodedCharacteristicSerializer