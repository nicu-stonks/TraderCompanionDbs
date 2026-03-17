from django.urls import path
from . import views

urlpatterns = [
    path('config/', views.LLMConfigView.as_view(), name='llm-config'),
    path('supplements/', views.SupplementListCreateView.as_view(), name='supplement-list-create'),
    path('supplements/<int:pk>/', views.SupplementDeleteView.as_view(), name='supplement-delete'),
    path('foods/', views.FoodItemListCreateView.as_view(), name='food-list-create'),
    path('foods/<int:pk>/', views.FoodItemDeleteView.as_view(), name='food-delete'),
    path('daily_records/', views.DailyRecordListView.as_view(), name='daily-record-list'),
    path('daily_records/<str:date>/', views.DailyRecordDetailView.as_view(), name='daily-record-detail'),
    path('chat/<str:date>/', views.ChatView.as_view(), name='chat'),
    path('chat/<str:date>/prompt/', views.ChatPromptView.as_view(), name='chat-prompt'),
    path('chat/<str:date>/manual/', views.ChatManualSubmitView.as_view(), name='chat-manual'),
    path('chat/<str:date>/rollback/', views.ChatRollbackView.as_view(), name='chat-rollback'),
]
