from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import MetricViewSet, TradeGradeViewSet, PostTradeAnalysisViewSet, MetricOptionRecommendationViewSet, MetricGradeCheckSettingViewSet, MetricPercentBaseSettingViewSet

router = DefaultRouter()
router.register(r'metrics', MetricViewSet)
router.register(r'grades', TradeGradeViewSet)
router.register(r'analyses', PostTradeAnalysisViewSet)
router.register(r'option-recommendations', MetricOptionRecommendationViewSet)
router.register(r'metric-check-settings', MetricGradeCheckSettingViewSet)
router.register(r'percent-base-settings', MetricPercentBaseSettingViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
]
