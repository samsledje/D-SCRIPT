from django.contrib import admin

from .models import Job


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = (
        "uuid",
        "title",
        "email",
        "seqsIndex",
        "pairsIndex",
        "seqs",
        "pairs",
        "completed",
    )
