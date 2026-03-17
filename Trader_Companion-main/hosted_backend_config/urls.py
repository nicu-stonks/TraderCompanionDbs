from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import HostedBackendCredentialsViewSet

router = DefaultRouter()
router.register(r'credentials', HostedBackendCredentialsViewSet, basename='hosted-credentials')

urlpatterns = [
    path('', include(router.urls)),
]
