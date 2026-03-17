from django.db import models


class HostedBackendCredentials(models.Model):
    """Model for storing hosted backend credentials"""
    username = models.CharField(max_length=100, unique=True)
    hosted_url = models.URLField(max_length=500)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_sync = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name_plural = "Hosted Backend Credentials"
    
    def __str__(self):
        return f"{self.username} @ {self.hosted_url}"
    
    @classmethod
    def get_active_credentials(cls):
        """Get the active credentials"""
        return cls.objects.filter(is_active=True).first()
