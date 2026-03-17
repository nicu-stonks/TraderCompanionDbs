from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'trades', views.TradesViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('balance/', views.balance_view, name='balance'),
]