import logging
from datetime import datetime

import pandas as pd
from celery import shared_task

from .api import dscript as dscript_api
from .models import Job


@shared_task(track_started=True)
def process_job(uuid):
    logging.info(uuid)
    job = Job.objects.get(pk=uuid)

    job.is_running = True
    job.start_time = datetime.utcnow()
    job.task_status = "STARTED"
    job.save()

    results_file = dscript_api.predict_pairs(job.uuid, job.seq_fi, job.pair_fi)
    try:
        logging.info("Trying to email")
        dscript_api.email_results(
            job.email, results_file, job.uuid, title=job.title
        )
    except Exception as err:
        logging.info("Not a valid email")
        logging.info(err)

    job.is_running = False
    job.is_completed = True
    job.task_status = "SUCCESS"
    res_df = pd.read_csv(results_file, sep="\t", header=None)
    job.n_pairs_done = len(res_df)
    job.save()

    return results_file
