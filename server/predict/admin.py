from django.contrib import admin

from .models import Job


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = (
        "uuid",
        "title",
        "email",
        "seq_fi",
        "pair_fi",
        "n_seqs",
        "n_pairs",
        "submission_time",
        "start_time",
        "queue_pos",
        "n_pairs_done",
        "is_running",
        "is_completed",
    )
