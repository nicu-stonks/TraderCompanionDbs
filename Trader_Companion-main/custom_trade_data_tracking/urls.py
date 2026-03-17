from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'columns', views.CustomColumnViewSet)
router.register(r'column_order', views.ColumnOrderViewSet)

urlpatterns = [
    # Manual paths must come BEFORE router.urls so they aren't caught by the router's <pk> patterns
    path('column_order/bulk/', views.bulk_update_column_order, name='bulk-column-order'),
    path('values/', views.get_custom_column_values, name='custom-column-values'),
    path('values/bulk/', views.bulk_upsert_custom_column_values, name='bulk-upsert-values'),
    path('values/by_column/<int:column_id>/', views.delete_custom_column_values_by_column, name='delete-values-by-column'),
    path('', include(router.urls)),
]
