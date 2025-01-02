from django.db import models
from django.contrib.auth.models import User

class AIModel(models.Model):
    MODEL_TYPE_CHOICES = [
        ('LSTM', 'Long Short-Term Memory'),
        ('Transformer', 'Transformer'),
        ('Q-Learning', 'Q-Learning'),
    ]
    
    name = models.CharField(max_length=50)
    description = models.TextField()
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    date_created = models.DateTimeField(auto_now_add=True)
    hyperparameters = models.JSONField(default=dict)
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"

class Agent(models.Model):
    name = models.CharField(max_length=50)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='agents')
    ai_model = models.ForeignKey(AIModel, on_delete=models.CASCADE)
    risk_tolerance = models.FloatField()
    investment_goal = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} - {self.user.username}"

class Backtest(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='backtests')
    start_date = models.DateField()
    end_date = models.DateField()
    initial_capital = models.FloatField()
    final_capital = models.FloatField()
    roi = models.FloatField()
    strategy_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Backtest ({self.agent.name}): {self.start_date} - {self.end_date}"

# 4. Report 선언
class Report(models.Model):
    REPORT_TYPE_CHOICES = [
        ('WEEKLY', 'Weekly'),
        ('MONTHLY', 'Monthly'),
        ('CUSTOM', 'Custom'),
    ]
    
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='reports')
    report_type = models.CharField(max_length=20, choices=REPORT_TYPE_CHOICES)
    performance_summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.report_type} Report - {self.agent.name}"
