from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import requests
import json
from django.conf import settings

FLASK_SERVICE_URL = settings.MICROSERVICE_SETTINGS['FLASK_SERVICE_URL']

def handle_microservice_error(error):
    """Helper function to handle microservice connection errors"""
    return JsonResponse({
        "status": "error",
        "error": f"Could not connect to microservice: {str(error)}"
    }, status=500)

@require_http_methods(["GET"])
def rankings_view(request, filename):
    """Dynamically forward ANY /rankings/... request to Flask"""
    try:
        flask_url = f"{FLASK_SERVICE_URL}/rankings/{filename}"
        response = requests.get(flask_url, params=request.GET)  # Forward query params too
        return JsonResponse(response.json(), status=response.status_code)
    except requests.RequestException as e:
        return handle_microservice_error(e)


@require_http_methods(["GET"])
def pipeline_status_view(request):
    """View to get the current pipeline status"""
    try:
        response = requests.get(f'{FLASK_SERVICE_URL}/pipeline/status')
        return JsonResponse(response.json(), status=response.status_code)
    except requests.RequestException as e:
        return handle_microservice_error(e)

@csrf_exempt
@require_http_methods(["POST"])
def screen_stocks_view(request):
    """View to initiate stock screening"""
    try:
        data = json.loads(request.body)
        response = requests.post(
            f'{FLASK_SERVICE_URL}/run_screening',
            json=data
        )
        return JsonResponse(response.json(), status=response.status_code)
    except requests.RequestException as e:
        return handle_microservice_error(e)
    except json.JSONDecodeError:
        return JsonResponse({
            "status": "error",
            "error": "Invalid JSON in request body"
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def ban_stocks_view(request):
    """View to ban stocks"""
    try:
        data = json.loads(request.body)
        response = requests.post(
            f'{FLASK_SERVICE_URL}/ban',
            json=data
        )
        return JsonResponse(response.json(), status=response.status_code)
    except requests.RequestException as e:
        return handle_microservice_error(e)
    except json.JSONDecodeError:
        return JsonResponse({
            "status": "error",
            "error": "Invalid JSON in request body"
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def stop_screening_view(request):
    """View to stop stock screening"""
    try:
        response = requests.post(f'{FLASK_SERVICE_URL}/pipeline/stop')
        return JsonResponse(response.json(), status=response.status_code)
    except requests.RequestException as e:
        return handle_microservice_error(e)