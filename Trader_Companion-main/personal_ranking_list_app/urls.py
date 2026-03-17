from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'ranking-boxes', views.RankingBoxViewSet)
router.register(r'stock-picks', views.StockPickViewSet)
router.register(r'global-characteristics', views.GlobalCharacteristicViewSet)
router.register(r'user-page-state', views.UserPageStateViewSet)
router.register(r'ordered-characteristics', views.OrderedCharacteristicViewSet)
router.register(r'priority-characteristics', views.PriorityCharacteristicViewSet)
router.register(r'color-coded-characteristics', views.ColorCodedCharacteristicViewSet)

urlpatterns = [
    path('', include(router.urls)),
]