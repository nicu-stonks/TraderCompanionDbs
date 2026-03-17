from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Trades, Balance
from .serializers import TradesSerializer, BalanceSerializer


class TradesViewSet(viewsets.ModelViewSet):
    queryset = Trades.objects.all()
    serializer_class = TradesSerializer


@api_view(['GET', 'PUT'])
def balance_view(request):
    # Ensure there's always one balance object
    balance_obj, created = Balance.objects.get_or_create(id=1)

    if request.method == 'GET':
        serializer = BalanceSerializer(balance_obj)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = BalanceSerializer(balance_obj, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)