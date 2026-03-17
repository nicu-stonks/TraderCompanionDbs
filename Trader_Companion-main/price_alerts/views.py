import logging

from rest_framework import status, viewsets
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from django.utils import timezone
from .models import Alert, AlarmSettings
from .serializers import AlertSerializer, AlarmSettingsSerializer
from .monitor import stop_alarm_playback

logger = logging.getLogger(__name__)


def _add_ticker_to_monitoring(ticker: str):
    """Add a ticker to the new ticker_data monitoring list via TrackedTicker model.
    Also fires an instant HTTP fetch to ensure data is immediately available.
    
    Returns: (success: bool, error_message: str or None)
    """
    from ticker_data.models import TrackedTicker
    try:
        logger.info(f"[TICKER] Adding {ticker} to monitoring")
        TrackedTicker.objects.get_or_create(symbol=ticker.upper().strip())
        return True, None
    except Exception as e:
        error_msg = f"Error adding ticker: {str(e)}"
        logger.error(f"[TICKER] {error_msg}")
        return False, error_msg


def _fetch_current_price(ticker: str):
    """Fetch current price and related data from the new local ticker_data Django DB.
    
    Returns tuple: (data: dict or None, error_message: str or None)
    
    data contains: {'price': float, 'previous_close': float or None, 'percent_change': float or None}
    """
    from ticker_data.models import HistoricalPrice
    try:
        logger.info(f"[FETCH PRICE] Getting cached price for {ticker} from local DB")
        # We need the 2 most recent days to calculate percent change
        recent_bars = HistoricalPrice.objects.filter(symbol=ticker.upper().strip()).order_by('-date')[:2]
        
        if len(recent_bars) > 0:
            latest = recent_bars[0]
            price = latest.close
            
            previous_close = None
            if len(recent_bars) > 1:
                previous_close = recent_bars[1].close
            
            percent_change = None
            if previous_close and previous_close > 0:
                percent_change = ((price - previous_close) / previous_close) * 100
            
            logger.info(f"[FETCH PRICE] {ticker} price: {price}, prev_close: {previous_close}, pct_change: {percent_change}")
            return {
                'price': float(price),
                'previous_close': float(previous_close) if previous_close else None,
                'percent_change': round(percent_change, 2) if percent_change is not None else None
            }, None
        else:
            error_msg = f"No price data available for {ticker} yet in DB."
            logger.warning(f"[FETCH PRICE] {error_msg}")
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error fetching price: {str(e)}"
        logger.error(f"[FETCH PRICE] {error_msg}")
        return None, error_msg



class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.all()
    serializer_class = AlertSerializer
    
    def create(self, request, *args, **kwargs):
        logger.info(f"[CREATE ALERT] Raw request data: {request.data}")
        logger.info(f"[CREATE ALERT] Request content-type: {request.content_type}")
        
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"[CREATE ALERT] Serializer validation failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ticker = serializer.validated_data['ticker'].upper()
        alert_price = serializer.validated_data['alert_price']
        logger.info(f"[CREATE ALERT] Validated data - ticker: {ticker}, alert_price: {alert_price}")

        # First, add ticker to monitoring (joins the round-robin loop)
        success, add_error = _add_ticker_to_monitoring(ticker)
        if not success:
            logger.error(f"[CREATE ALERT] Failed to add ticker to monitoring: {add_error}")
            return Response(
                {'error': add_error or f'Could not add {ticker} to monitoring'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create alert with null price - the monitor will update it on the next check cycle
        logger.info(f"[CREATE ALERT] Creating alert for {ticker} (price will be fetched by monitor)")
        
        alert = serializer.save(
            ticker=ticker,
            current_price=None,
            initial_price_above_alert=None,
            last_checked=None,
            previous_close=None,
            percent_change=None
        )

        return Response(AlertSerializer(alert).data, status=status.HTTP_201_CREATED)
    
    def update(self, request, *args, **kwargs):
        """Update alert and re-fetch price if needed"""
        instance = self.get_object()
        prev_is_active = instance.is_active
        prev_triggered = instance.triggered

        print(f"[VIEWS UPDATE] Alert ID: {instance.id}, prev_is_active={prev_is_active}, prev_triggered={prev_triggered}")
        print(f"[VIEWS UPDATE] Request data: {request.data}")

        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)

        requested_is_active = serializer.validated_data.get('is_active')
        print(f"[VIEWS UPDATE] requested_is_active={requested_is_active}")
        
        # If user is stopping the alert (setting is_active=False), handle it immediately
        if requested_is_active is False:
            print(f"[VIEWS UPDATE] User stopping alert, prev_is_active={prev_is_active}")
            if not prev_is_active:
                # Already stopped, just return current state
                print("[VIEWS UPDATE] Alert already stopped, returning")
                return Response(AlertSerializer(instance).data)
            
            # Stop the alert - FORCE STOP ALARM IMMEDIATELY
            print(f"[VIEWS UPDATE] CALLING stop_alarm_playback for alert {instance.id}")
            stop_alarm_playback(instance.id)  # Stop alarm FIRST before saving
            instance.is_active = False
            instance.triggered = False
            instance.triggered_at = None
            instance.initial_price_above_alert = None
            instance.save(update_fields=['is_active', 'triggered', 'triggered_at', 'initial_price_above_alert'])
            # Stop again after save to be absolutely sure
            stop_alarm_playback(instance.id)
            return Response(AlertSerializer(instance).data)
        
        # Prevent reactivation of stopped alerts
        if requested_is_active is True and not prev_is_active:
            return Response(
                {'error': 'Stopped alerts cannot be reactivated. Please create a new alert.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        alert = serializer.save()

        if alert.is_active and (not prev_is_active or prev_triggered or 'alert_price' in serializer.validated_data):
            # Only fetch price if alert is being reactivated or price changed
            current_price, error_msg = _fetch_current_price(alert.ticker)
            if current_price is not None:
                alert.current_price = current_price
                alert.last_checked = timezone.now()
                alert.initial_price_above_alert = current_price > alert.alert_price
            else:
                alert.initial_price_above_alert = None
                alert.last_checked = timezone.now()
            alert.triggered = False
            alert.triggered_at = None
            alert.save(update_fields=[
                'current_price', 'last_checked', 'initial_price_above_alert',
                'triggered', 'triggered_at'
            ])
        elif not alert.is_active and prev_is_active:
            # Alert is being deactivated - stop fetching, keep last known price
            alert.triggered = False
            alert.triggered_at = None
            alert.initial_price_above_alert = None
            # Don't update current_price or last_checked - keep last known values
            alert.save(update_fields=['triggered', 'triggered_at', 'initial_price_above_alert'])
            stop_alarm_playback(alert.id)
            alert.triggered = False
            alert.triggered_at = None
            alert.initial_price_above_alert = None
            alert.save(update_fields=['triggered', 'triggered_at', 'initial_price_above_alert'])
            stop_alarm_playback(alert.id)

        return Response(AlertSerializer(alert).data)

    def destroy(self, request, *args, **kwargs):
        alert = self.get_object()
        
        # Stop any playing alarm before deleting
        if alert.is_active or alert.triggered:
            stop_alarm_playback(alert.id)
        
        return super().destroy(request, *args, **kwargs)
    
    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        """Delete all alerts at once."""
        alerts = Alert.objects.all()
        count = alerts.count()
        
        # Stop all playing alarms first
        for alert in alerts:
            if alert.is_active or alert.triggered:
                stop_alarm_playback(alert.id)
        
        # Delete all alerts
        alerts.delete()
        
        return Response({
            'message': f'Successfully deleted {count} alert(s)',
            'deleted_count': count
        }, status=status.HTTP_200_OK)


@api_view(['GET', 'PUT'])
def alarm_settings_view(request):
    """Get or update alarm settings"""
    settings_obj = AlarmSettings.get_settings()
    
    if request.method == 'GET':
        serializer = AlarmSettingsSerializer(settings_obj)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = AlarmSettingsSerializer(settings_obj, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def upload_alarm_sound(request):
    """Upload a custom alarm sound file"""
    if 'file' not in request.FILES:
        return Response(
            {'error': 'No file provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    file = request.FILES['file']
    
    # Validate file type
    if not file.name.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
        return Response(
            {'error': 'Invalid file type. Only audio files (mp3, wav, ogg, m4a) are allowed.'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Save file to alarm_sounds directory
    import os
    from django.conf import settings
    
    alarm_sounds_dir = os.path.join(settings.BASE_DIR, 'alarm_sounds')
    os.makedirs(alarm_sounds_dir, exist_ok=True)
    
    file_path = os.path.join(alarm_sounds_dir, file.name)
    
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    return Response({
        'message': 'File uploaded successfully',
        'filename': file.name,
        'path': file.name  # Just return filename, frontend will construct path
    })


@api_view(['GET'])
def list_alarm_sounds(request):
    """List all available alarm sound files"""
    import os
    from django.conf import settings
    
    alarm_sounds_dir = os.path.join(settings.BASE_DIR, 'alarm_sounds')
    sounds = []
    
    if os.path.exists(alarm_sounds_dir):
        for filename in os.listdir(alarm_sounds_dir):
            if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                sounds.append(filename)
    
    return Response({'sounds': sorted(sounds)})


@api_view(['GET'])
def serve_alarm_sound(request, filename):
    """Serve an alarm sound file"""
    import os
    from django.conf import settings
    from django.http import FileResponse, Http404
    from django.views.decorators.cache import cache_control
    
    alarm_sounds_dir = os.path.join(settings.BASE_DIR, 'alarm_sounds')
    file_path = os.path.join(alarm_sounds_dir, filename)
    
    # Security: ensure file is within the alarm_sounds directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(alarm_sounds_dir)):
        raise Http404("Invalid file path")
    
    if not os.path.exists(file_path):
        raise Http404(f"Sound file not found: {filename}")
    
    # Determine content type
    content_type_map = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
    }
    ext = os.path.splitext(filename)[1].lower()
    content_type = content_type_map.get(ext, 'audio/mpeg')
    
    response = FileResponse(open(file_path, 'rb'), content_type=content_type)
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    # Enable CORS for audio files
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'GET'
    return response


@api_view(['POST'])
def stop_alarm_view(request, alert_id=None):
    """Stop alarm playback for a specific alert or all alarms.
    
    Args:
        alert_id: Optional alert ID from URL. If provided, stops only that alarm.
                 If None, stops all alarms.
    """
    try:
        if alert_id is not None:
            # Verify the alert exists
            try:
                Alert.objects.get(id=alert_id)
            except Alert.DoesNotExist:
                return Response(
                    {'error': f'Alert with ID {alert_id} not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            stop_alarm_playback(alert_id)
            return Response({
                'message': f'Alarm stopped for alert {alert_id}',
                'alert_id': alert_id
            })
        else:
            stop_alarm_playback()
            return Response({
                'message': 'All alarms stopped'
            })
    except Exception as e:
        logger.error(f"Error stopping alarm: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============ Telegram Notification Endpoints ============

@api_view(['GET'])
def get_telegram_config(request):
    """Get current Telegram configuration"""
    try:
        from .models import TelegramConfig
        config = TelegramConfig.get_config()
        
        return Response({
            'bot_token': config.bot_token,
            'chat_id': config.chat_id,
            'enabled': config.enabled,
            'configured': bool(config.bot_token and config.chat_id)
        })
    except Exception as e:
        logger.error(f"Error getting Telegram config: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def save_telegram_config(request):
    """Save or update Telegram configuration"""
    try:
        from .models import TelegramConfig
        
        bot_token = request.data.get('bot_token', '').strip()
        chat_id = request.data.get('chat_id', '').strip()
        
        if not bot_token or not chat_id:
            return Response(
                {'error': 'Both bot_token and chat_id are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        config = TelegramConfig.get_config()
        config.bot_token = bot_token
        config.chat_id = chat_id
        config.save()
        
        return Response({
            'message': 'Telegram configuration saved successfully',
            'bot_token': config.bot_token,
            'chat_id': config.chat_id,
            'enabled': config.enabled
        })
    except Exception as e:
        logger.error(f"Error saving Telegram config: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def test_telegram_connection(request):
    """Test Telegram bot connection"""
    try:
        from . import telegram_notifier
        
        bot_token = request.data.get('bot_token', '').strip()
        chat_id = request.data.get('chat_id', '').strip()
        
        if not bot_token or not chat_id:
            return Response(
                {'error': 'Both bot_token and chat_id are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = telegram_notifier.test_telegram_connection(bot_token, chat_id)
        
        if result['success']:
            return Response(result)
        else:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"Error testing Telegram connection: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def toggle_telegram_notifications(request):
    """Enable or disable Telegram notifications"""
    try:
        from .models import TelegramConfig
        
        enabled = request.data.get('enabled')
        
        if enabled is None:
            return Response(
                {'error': 'enabled field is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        config = TelegramConfig.get_config()
        
        # Verify configuration exists if trying to enable
        if enabled and (not config.bot_token or not config.chat_id):
            return Response(
                {'error': 'Cannot enable notifications without bot_token and chat_id configured'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        config.enabled = bool(enabled)
        config.save()
        
        return Response({
            'message': f'Telegram notifications {"enabled" if config.enabled else "disabled"}',
            'enabled': config.enabled
        })
    except Exception as e:
        logger.error(f"Error toggling Telegram notifications: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
