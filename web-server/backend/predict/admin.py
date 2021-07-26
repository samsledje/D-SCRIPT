from django.contrib import admin
from .models import SinglePair, ManyPair, AllPair
from .models import PairsUpload, PairsInput, SeqsUpload, SeqsInput, PredictionJob

class SinglePairAdmin(admin.ModelAdmin):
    # Displays within admin panel
    list_display = ('title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

class ManyPairAdmin(admin.ModelAdmin):
    list_display = ('title', 'pairs', 'sequences', 'predictions')

class AllPairAdmin(admin.ModelAdmin):
    list_display = ('title', 'sequences', 'predictions')


class PairsUploadAdmin(admin.ModelAdmin):
    list_display = ('pairs',)

class PairsInputAdmin(admin.ModelAdmin):
    list_display = ('pairs',)

class SeqsUploadAdmin(admin.ModelAdmin):
    list_display = ('seqs',)

class SeqsInputAdmin(admin.ModelAdmin):
    list_display = ('seqs',)

class PredictionJobAdmin(admin.ModelAdmin):
    list_display = ()



# Register your models here.

admin.site.register(SinglePair, SinglePairAdmin)
admin.site.register(ManyPair, ManyPairAdmin)
admin.site.register(AllPair, AllPairAdmin)

admin.site.register(PairsUpload, PairsUploadAdmin)
admin.site.register(PairsInput, PairsInputAdmin)
admin.site.register(SeqsUpload, SeqsUploadAdmin)
admin.site.register(SeqsInput, SeqsInputAdmin)
admin.site.register(PredictionJob, PredictionJobAdmin)