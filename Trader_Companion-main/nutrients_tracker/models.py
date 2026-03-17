from django.db import models

class LLMConfig(models.Model):
    model_name = models.CharField(max_length=255, default="gemini-3-flash-preview")
    api_key = models.CharField(max_length=500, blank=True, null=True)
    user_profile = models.TextField(blank=True, default="", help_text="Age, weight, daily activities, etc.")
    available_models = models.TextField(blank=True, default="gemini-3-flash-preview,gemini-1.5-pro-preview-0409", help_text="Comma-separated list of models")
    
    # We'll just enforce a single row for config
    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

class Supplement(models.Model):
    name = models.CharField(max_length=255, unique=True)
    details = models.TextField(blank=True, null=True, help_text="What it contains, doses, etc.")

    def __str__(self):
        return self.name

class FoodItem(models.Model):
    name = models.CharField(max_length=255, unique=True)
    details = models.TextField(blank=True, null=True, help_text="Macros, portions, etc.")

    def __str__(self):
        return self.name

class DailyRecord(models.Model):
    date = models.DateField(primary_key=True)
    foods_eaten = models.TextField(blank=True, default="")
    supplements_taken = models.JSONField(default=list, blank=True)
    
    # The percentage fills for nutrients as decided by the LLM
    # E.g., {"Protein": 80, "Vitamin C": 100, "Omega 3": 50}
    nutrient_completion = models.JSONField(default=dict, blank=True)
    
    # The dietary sources that contribute to each nutrient
    # E.g., {"Protein": ["500g carbonara", "5 eggs"], "Vitamin C": ["oranges"]}
    nutrient_sources = models.JSONField(default=dict, blank=True)
    
    # Text or JSON indicating what the LLM recommends the user still take today
    recommendations = models.TextField(blank=True, default="")
    
    # Conversation history to allow follow-up prompting
    # E.g., [{"role": "user", "parts": ["..."]}, {"role": "model", "parts": ["..."]}]
    chat_history = models.JSONField(default=list, blank=True)

    def __str__(self):
        return str(self.date)
