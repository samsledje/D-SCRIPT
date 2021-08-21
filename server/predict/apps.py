import sys

from django.apps import AppConfig


class PredictConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predict"

    def ready(self):
        if "runserver" not in sys.argv:
            return True
        # you must import your modules here
        # to avoid AppRegistryNotReady exception
        from .tasks import sweep_incomplete_jobs

        rslt = sweep_incomplete_jobs.delay()
        rslt.get()
