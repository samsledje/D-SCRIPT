from django.contrib import admin

from .models import Job


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


# Register your models here.

admin.site.register(Job, JobAdmin)
