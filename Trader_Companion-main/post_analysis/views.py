from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db import transaction
from .models import Metric, MetricOption, TradeGrade, PostTradeAnalysis, MetricOptionRecommendation, MetricGradeCheckSetting, MetricPercentBaseSetting
from .serializers import (
    MetricSerializer, CreateMetricSerializer, MetricOptionSerializer,
    TradeGradeSerializer, BulkTradeGradeSerializer, PostTradeAnalysisSerializer,
    MetricOptionRecommendationSerializer, MetricGradeCheckSettingSerializer, MetricPercentBaseSettingSerializer
)


class MetricViewSet(viewsets.ModelViewSet):
    queryset = Metric.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return CreateMetricSerializer
        return MetricSerializer
    
    @action(detail=True, methods=['post'])
    def add_option(self, request, pk=None):
        """Add a new option to an existing metric"""
        metric = self.get_object()
        option_name = request.data.get('name')
        
        if not option_name:
            return Response(
                {'error': 'Option name is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get the next value for ordering
        last_option = metric.options.order_by('-value').first()
        next_value = (last_option.value + 1) if last_option else 0
        
        option = MetricOption.objects.create(
            metric=metric,
            name=option_name,
            value=next_value
        )
        
        serializer = MetricOptionSerializer(option)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['delete'], url_path='options/(?P<option_id>[^/.]+)')
    def remove_option(self, request, pk=None, option_id=None):
        """Remove an option from a metric"""
        metric = self.get_object()
        try:
            option = metric.options.get(id=option_id)
            option.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except MetricOption.DoesNotExist:
            return Response(
                {'error': 'Option not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )


class TradeGradeViewSet(viewsets.ModelViewSet):
    queryset = TradeGrade.objects.all()
    serializer_class = TradeGradeSerializer
    
    def get_queryset(self):
        """Filter by trade_id if provided"""
        queryset = super().get_queryset()
        trade_id = self.request.query_params.get('trade_id')
        if trade_id:
            queryset = queryset.filter(trade_id=trade_id)
        return queryset
    
    @action(detail=False, methods=['post'])
    def bulk_update(self, request):
        """Bulk update trade grades"""
        serializer = BulkTradeGradeSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        grades_data = serializer.validated_data['grades']
        deletions_data = serializer.validated_data.get('deletions', [])

        with transaction.atomic():
            # Process deletions first so re-added grades in same payload work
            for deletion in deletions_data:
                trade_id = int(deletion['tradeId'])
                metric_id = int(deletion['metricId'])
                TradeGrade.objects.filter(trade_id=trade_id, metric_id=metric_id).delete()

            for grade_data in grades_data:
                trade_id = int(grade_data['tradeId'])
                metric_id = int(grade_data['metricId'])
                selected_option_id = int(grade_data['selectedOptionId'])

                # Validate that metric and option exist
                try:
                    metric = Metric.objects.get(id=metric_id)
                    option = MetricOption.objects.get(id=selected_option_id, metric=metric)
                except (Metric.DoesNotExist, MetricOption.DoesNotExist):
                    return Response(
                        {'error': f'Invalid metric or option for grade: {grade_data}'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Update or create the grade
                TradeGrade.objects.update_or_create(
                    trade_id=trade_id,
                    metric_id=metric_id,
                    defaults={'selected_option_id': selected_option_id}
                )
        
        # Return all grades for confirmation
        all_grades = TradeGrade.objects.all()
        return Response(
            TradeGradeSerializer(all_grades, many=True).data,
            status=status.HTTP_200_OK
        )
    
    @action(detail=False, methods=['get'])
    def analytics_data(self, request):
        """Get analytics data for charting"""
        # This endpoint will be used by the frontend to get data for charts
        # You can implement specific analytics logic here
        
        grades = self.get_queryset().select_related('metric', 'selected_option')
        
        # Group by metric for easier processing
        analytics_data = {}
        for grade in grades:
            metric_name = grade.metric.name
            if metric_name not in analytics_data:
                analytics_data[metric_name] = []
            
            analytics_data[metric_name].append({
                'trade_id': grade.trade_id,
                'option': grade.selected_option.name,
                'graded_at': grade.graded_at.isoformat()
            })
        
        return Response(analytics_data)


class PostTradeAnalysisViewSet(viewsets.ModelViewSet):
    queryset = PostTradeAnalysis.objects.all()
    serializer_class = PostTradeAnalysisSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        trade_id = self.request.query_params.get('trade_id')
        if trade_id:
            qs = qs.filter(trade_id=trade_id)
        return qs

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user if self.request.user.is_authenticated else None)


class MetricOptionRecommendationViewSet(viewsets.ModelViewSet):
    queryset = MetricOptionRecommendation.objects.all()
    serializer_class = MetricOptionRecommendationSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        metric_id = serializer.validated_data['metric_id']
        option_id = serializer.validated_data['option_id']
        recommended_pct = serializer.validated_data['recommended_pct']
        is_minimum = serializer.validated_data.get('is_minimum', True)

        recommendation, created = MetricOptionRecommendation.objects.update_or_create(
            metric_id=metric_id,
            option_id=option_id,
            defaults={'recommended_pct': recommended_pct, 'is_minimum': is_minimum}
        )

        output = self.get_serializer(recommendation)
        status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(output.data, status=status_code)


class MetricGradeCheckSettingViewSet(viewsets.ModelViewSet):
    queryset = MetricGradeCheckSetting.objects.using('post_analysis_recommendations').all()
    serializer_class = MetricGradeCheckSettingSerializer

    DEFAULT_REQUIRED = 'Exit at loss, Exit at profit'
    DEFAULT_EXCLUDE = 'Entry Point'

    def get_queryset(self):
        return MetricGradeCheckSetting.objects.using('post_analysis_recommendations').all()

    def list(self, request, *args, **kwargs):
        qs = self.get_queryset()
        if not qs.exists():
            MetricGradeCheckSetting.objects.using('post_analysis_recommendations').create(
                required_metrics=self.DEFAULT_REQUIRED,
                exclude_metric=self.DEFAULT_EXCLUDE
            )
            qs = self.get_queryset()
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        required_metrics = serializer.validated_data.get('required_metrics', '')
        exclude_metric = serializer.validated_data.get('exclude_metric', '')

        setting, created = MetricGradeCheckSetting.objects.using('post_analysis_recommendations').update_or_create(
            id=1,
            defaults={
                'required_metrics': required_metrics,
                'exclude_metric': exclude_metric
            }
        )

        output = self.get_serializer(setting)
        status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(output.data, status=status_code)


class MetricPercentBaseSettingViewSet(viewsets.ModelViewSet):
    queryset = MetricPercentBaseSetting.objects.using('post_analysis_recommendations').all()
    serializer_class = MetricPercentBaseSettingSerializer

    def get_queryset(self):
        return MetricPercentBaseSetting.objects.using('post_analysis_recommendations').all()

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        metric_id = serializer.validated_data['metric_id']
        use_total_trades = serializer.validated_data.get('use_total_trades', False)

        setting, created = MetricPercentBaseSetting.objects.using('post_analysis_recommendations').update_or_create(
            metric_id=metric_id,
            defaults={'use_total_trades': use_total_trades}
        )
        output = self.get_serializer(setting)
        status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(output.data, status=status_code)

