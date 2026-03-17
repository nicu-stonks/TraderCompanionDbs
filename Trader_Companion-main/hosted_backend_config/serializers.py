from rest_framework import serializers
from .models import HostedBackendCredentials


class HostedBackendCredentialsSerializer(serializers.ModelSerializer):
    class Meta:
        model = HostedBackendCredentials
        fields = ['id', 'username', 'hosted_url', 'is_active', 'created_at', 'updated_at', 'last_sync']
        read_only_fields = ['id', 'created_at', 'updated_at', 'last_sync']
