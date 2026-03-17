"""
URL configuration for djangoProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path('stock_filtering_app/', include('Stocks_Filtering_App.urls')),
    path('personal_ranking/', include('personal_ranking_list_app.urls')),
    path('trades_app/', include('trades_history.urls')),
    path('post_analysis/', include('post_analysis.urls')),
    path('price_alerts/', include('price_alerts.urls')),
    path('hosted_backend/', include('hosted_backend_config.urls')),
    path('custom_trade_data/', include('custom_trade_data_tracking.urls')),
    path('violations_monitor/', include('violations_monitor.urls')),
    path('violations_monalert/', include('violations_monitor.urls')),
    path('api/nutrients_tracker/', include('nutrients_tracker.urls')),
    path('ticker_data/', include('ticker_data.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # Serve alarm sounds in development
    from django.views.static import serve
    from django.urls import re_path
    import os
    
    alarm_sounds_dir = os.path.join(settings.BASE_DIR, 'alarm_sounds')
    urlpatterns += [
        re_path(r'^alarm_sounds/(?P<path>.*)$', serve, {'document_root': alarm_sounds_dir}),
    ]
