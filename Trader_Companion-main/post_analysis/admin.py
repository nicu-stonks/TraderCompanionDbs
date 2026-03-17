from django.contrib import admin
from .models import Metric, MetricOption, TradeGrade, PostTradeAnalysis, MetricOptionRecommendation


class MetricOptionInline(admin.TabularInline):
    model = MetricOption
    extra = 1


@admin.register(Metric)
class MetricAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at', 'updated_at']
    search_fields = ['name']
    inlines = [MetricOptionInline]


@admin.register(TradeGrade)
class TradeGradeAdmin(admin.ModelAdmin):
    list_display = ['trade_id', 'metric', 'selected_option', 'graded_at']
    list_filter = ['metric', 'graded_at']
    search_fields = ['trade_id']


@admin.register(PostTradeAnalysis)
class PostTradeAnalysisAdmin(admin.ModelAdmin):
    list_display = ['trade_id', 'title', 'created_by', 'updated_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['trade_id', 'title', 'notes']


@admin.register(MetricOptionRecommendation)
class MetricOptionRecommendationAdmin(admin.ModelAdmin):
    list_display = ['metric_id', 'option_id', 'recommended_pct', 'updated_at']
    list_filter = ['metric_id']
    search_fields = ['metric_id', 'option_id']

