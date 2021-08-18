import logging

from celery import shared_task

from .api import dscript as dscript_api
from .models import Job


@shared_task
def test_adding(x, y):
    return x + y


@shared_task(track_started=True)
def process_job(uuid):
    logging.info(uuid)
    job = Job.objects.get(pk=uuid)

    results_file = dscript_api.predict_pairs(job.uuid, job.seq_fi, job.pair_fi)
    try:
        logging.info("Trying to email")
        dscript_api.email_results(
            job.email, results_file, job.uuid, title=job.title
        )
    except Exception as err:
        logging.info("Not a valid email")
        logging.info(err)
    return results_file
