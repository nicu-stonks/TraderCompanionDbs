from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from .models import HostedBackendCredentials
from .serializers import HostedBackendCredentialsSerializer


class HostedBackendCredentialsViewSet(viewsets.ModelViewSet):
    queryset = HostedBackendCredentials.objects.all()
    serializer_class = HostedBackendCredentialsSerializer
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get the active credentials"""
        credentials = HostedBackendCredentials.get_active_credentials()
        if credentials:
            serializer = self.get_serializer(credentials)
            return Response(serializer.data)
        return Response({'detail': 'No active credentials found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def set_active(self, request, pk=None):
        """Set a credential as active (deactivates all others)"""
        # Deactivate all
        HostedBackendCredentials.objects.all().update(is_active=False)
        
        # Activate the selected one
        credential = self.get_object()
        credential.is_active = True
        credential.save()
        
        serializer = self.get_serializer(credential)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def update_sync_time(self, request, pk=None):
        """Update the last sync timestamp"""
        credential = self.get_object()
        credential.last_sync = timezone.now()
        credential.save()
        
        serializer = self.get_serializer(credential)
        return Response(serializer.data)
