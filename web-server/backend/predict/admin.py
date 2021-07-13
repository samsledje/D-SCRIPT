from django.contrib import admin
from .models import SinglePair, ManyPair

class SinglePairAdmin(admin.ModelAdmin):
    # Displays within admin panel
    list_display = ('title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

class ManyPairAdmin(admin.ModelAdmin):
    list_display = ('title', 'pairs', 'sequences', 'predictions')

# Register your models here.

admin.site.register(SinglePair, SinglePairAdmin)
admin.site.register(ManyPair, ManyPairAdmin)