from django.contrib import admin

from .models import Job


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = (
        "uuid",
        "title",
        "email",
        # "seq_fi",
        # "pair_fi",
        # "result_fi",
        "n_seqs",
        "n_pairs",
        "submission_time",
        "start_time",
        "n_pairs_done",
        "is_running",
        "is_completed",
        "task_status",
    )
    list_filter = (
        "is_running",
        "is_completed",
        "task_status",
        "submission_time",
    )
    ordering = ("-submission_time",)
    date_hierarchy = "submission_time"
    search_fields = ("uuid", "title", "email")
