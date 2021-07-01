from django.contrib import admin
from .models import Prediction
from .models import FilePrediction

class PredictionAdmin(admin.ModelAdmin):
    # Displays within admin panel
    list_display = ('title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

class FilePredictionAdmin(admin.ModelAdmin):
    list_display = ('title', 'pairs', 'sequences', 'predictions')

# Register your models here.

admin.site.register(Prediction, PredictionAdmin)
admin.site.register(FilePrediction, FilePredictionAdmin)