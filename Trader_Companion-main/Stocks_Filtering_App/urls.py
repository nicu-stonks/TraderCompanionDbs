from django.urls import path
from . import views

urlpatterns = [
    path('rankings/<path:filename>', views.rankings_view, name='rankings'),
    path('pipeline/status', views.pipeline_status_view, name='pipeline-status'),
    path('run_screening', views.screen_stocks_view, name='screen-stocks'),
    path('ban', views.ban_stocks_view, name='ban-stocks'),
    path('pipeline/stop', views.stop_screening_view, name='stop-screening'),
]