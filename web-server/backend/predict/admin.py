from django.contrib import admin
from .models import Prediction

class PredictionAdmin(admin.ModelAdmin):
    # Displays within admin panel
    list_display = ('title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

# Register your models here.

admin.site.register(Prediction, PredictionAdmin)