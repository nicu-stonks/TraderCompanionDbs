from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'trades', views.MonAlertTradeViewSet)

urlpatterns = [
    path('preferences/', views.preferences_view, name='violations-preferences'),
    path('chart-settings/<str:ticker>/', views.chart_settings_view, name='violations-chart-settings'),
    path('compute/<int:trade_id>/', views.compute_violations_view, name='violations-compute'),
    path('compute-session/<int:trade_id>/', views.compute_session_violations_view, name='violations-compute-session'),
    path('compute-all/', views.compute_all_violations_view, name='violations-compute-all'),
    path('compute-status/', views.compute_status_view, name='violations-compute-status'),
    path('compute-health/', views.compute_health_view, name='violations-compute-health'),
    path('refresh/', views.refresh_historical_data, name='violations-refresh'),
    path('historical/<str:ticker>/', views.historical_data_info, name='violations-historical-info'),
    path('', include(router.urls)),
]
