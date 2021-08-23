import logging
from datetime import datetime

import pandas as pd
from celery import shared_task
from django.conf import settings
from django.db.models import Q

from .api import dscript as dscript_api
from .models import Job


@shared_task(track_started=True)
def process_job(uuid):
    logging.info(f"Launching job {uuid}")
    job = Job.objects.get(pk=uuid)

    if settings.DSCRIPT_CONFIRM_SUBMISSION_EMAIL:
        logging.info(f"Sending confirmation email for {uuid}")
        try:
            dscript_api.email_confirmation(job.uuid)
        except Exception as err:
            logging.error(str(err))

    job.is_running = True
    job.start_time = datetime.utcnow()
    job.task_status = "STARTED"
    job.save()

    results_file = dscript_api.predict_pairs(job.uuid)
    try:
        logging.debug("Trying to email")
        dscript_api.email_results(job.uuid)
    except Exception as err:
        logging.error(err)

    job.is_running = False
    job.is_completed = True
    job.task_status = "SUCCESS"
    res_df = pd.read_csv(results_file, sep="\t", header=None)
    job.n_pairs_done = len(res_df)
    job.save()

    return


@shared_task
def sweep_incomplete_jobs():
    logging.info("Sweeping incomplete jobs into the queue...")
    hanging_jobs = Job.objects.filter(
        Q(task_status="PENDING") | Q(task_status="STARTED")
    )
    for j in hanging_jobs:
        logging.info(f"Queuing job {j.uuid}")
        process_job.delay(j.uuid)
