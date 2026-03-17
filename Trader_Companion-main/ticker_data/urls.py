
from django.urls import path
from .views import InstantFetchView, provider_settings_view, request_stats_view, ticker_errors_view, tickers_view, request_interval_view
from .views import get_webull_status, start_webull_login, get_all_latest_data, get_latest_data, get_server_status
from .views import get_historical_data, get_historical_5m_data, get_historical_weekly_data, purge_all_price_data

urlpatterns = [
    path('api/ticker_data/status', get_server_status, name='server_status'),
    path('api/ticker_data/historical/<str:symbol>', get_historical_data, name='historical_data'),
    path('api/ticker_data/historical_weekly/<str:symbol>', get_historical_weekly_data, name='historical_weekly_data'),
    path('api/ticker_data/historical_5m/<str:symbol>', get_historical_5m_data, name='historical_5m_data'),
    path('api/ticker_data/fetch_now', InstantFetchView.as_view(), name='fetch_now'),
    path('api/ticker_data/settings', provider_settings_view, name='provider_settings'),
    path('api/ticker_data/request-stats', request_stats_view, name='request_stats'),
    path('api/ticker_data/errors', ticker_errors_view, name='ticker_errors'),
    path('api/ticker_data/tickers', tickers_view, name='tickers'),
    path('api/ticker_data/tickers/<str:symbol>', tickers_view, name='tickers_symbol'),
    path('api/ticker_data/request-interval', request_interval_view, name='request_interval'),
    
    path('api/ticker_data/webull/status', get_webull_status, name='webull_status'),
    path('api/ticker_data/webull/login', start_webull_login, name='webull_login'),
    path('api/ticker_data/webull/retry', start_webull_login, name='webull_retry'),
    path('api/ticker_data/data/latest/all', get_all_latest_data, name='latest_data_all'),
    path('api/ticker_data/data/<str:symbol>/latest', get_latest_data, name='latest_data_symbol'),
    path('api/ticker_data/purge-all-price-data', purge_all_price_data, name='purge_all_price_data'),
]
