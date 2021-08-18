import itertools
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpResponse
from django.views.generic import View
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from dscript.fasta import parse_input

from .api import dscript as dscript_api
from .models import Job
from .serializers import JobSerializer
from .tasks import process_job


class FrontendAppView(View):
    """
    Serves the compiled frontend entry point (only works if you have run `npm
    run build`).
    """

    def get(self, request, *args, **kwargs):
        logging.info(
            os.path.join(settings.REACT_APP_DIR, "build", "index.html")
        )
        try:
            with open(
                os.path.join(settings.REACT_APP_DIR, "build", "index.html")
            ) as f:
                return HttpResponse(f.read())
        except FileNotFoundError:
            logging.exception("Production build of app not found")
            return HttpResponse(
                """
                    This URL is only used when you have built the production
                    version of the app. Visit http://localhost:3000/ instead, or
                    run `yarn run build` to test the production version.
                    """,
                status=501,
            )


async_job_dict = {}


def upload_stream_to_local(in_file, out_file):
    with open(out_file, "w+", newline="\n") as out_f:
        if isinstance(in_file, UploadedFile):
            for chunk in in_file.chunks():
                content = chunk.decode("utf-8").strip().replace(",", "\t")
                out_f.write(content)
        elif isinstance(in_file, str):
            out_f.write(in_file.replace(",", "\t"))
    return out_file


def get_all_pairs(seq_file):
    with open(seq_file, "r") as f:
        nam, _ = parse_input(f.read())
        pairs = "\n".join("\t".join(p) for p in itertools.combinations(nam, 2))
        return pairs


@api_view(["GET", "POST"])
def predict(request):
    """
    Given a prediction input, queues a job for the prediction
    Returns the job id and whether the job comes first to the user
    """
    if request.method == "POST":
        data = request.data
        job_id = uuid.uuid4()

        seqs_upload = data["seqs"]
        pairs_upload = data["pairs"]

        seq_path = upload_stream_to_local(
            seqs_upload, f"{tempfile.gettempdir()}/{job_id}.fasta"
        )
        if int(data["pairsIndex"]) == 3:
            pairs_upload = get_all_pairs(seq_path)
        pair_path = upload_stream_to_local(
            pairs_upload, f"{tempfile.gettempdir()}/{job_id}.tsv"
        )

        logging.info(seq_path)
        logging.info(pair_path)
        logging.debug("seqs:")
        with open(seq_path, "r") as f:
            logging.debug(f.read())
        logging.debug("pairs:")
        with open(pair_path, "r") as f:
            logging.debug(f.read())

        job_data = {
            "uuid": job_id,
            "title": data["title"],
            "email": data["email"],
            "seq_fi": seq_path,
            "pair_fi": pair_path,
            "n_seqs": 0,
            "n_pairs": 0,
            "submission_time": datetime.utcnow(),
            "n_pairs_done": 0,
            "is_running": False,
            "is_completed": False,
        }

        job_m = Job(**job_data)
        job_m.save()
        job_async = process_job.delay(job_m.uuid)
        async_job_dict[job_id] = job_async

        response = {"id": job_m.uuid, "first": job_m.is_running}
        return Response(response)


@api_view(["GET"])
def get_position(request, id):
    logging.info(f" # Getting Queue Position for {id} ...")

    job_async = async_job_dict[id]
    job_state = job_async.state

    logging.debug(f"Job {id} status {job_state}")
    logging.info("# Sending response")
    return Response({"id": id, "status": job_state})

    # if job_state == "PENDING":
    #     return Response({
    #         "id": id,
    #         "status": job_state
    #         })
    # elif job_state == "STARTED":
    #     return Response({"position": 0, "inQueue": False})
    # elif job_state == "SUCCESS":
    #     rslt = job_async.get()
    #     return Response({"position": -1, "inQueue": False})
    # elif job_state == "FAILURE":
    #     rslt = job_async.get()
    #     return Response({"position": -1, "inQueue": False})


@api_view(["POST"])
def process_jobs(request):
    run_jobs()
    return Response({})


def run_jobs():
    pass
    # job = jobs[0]
    # logging.info(f" # Processing Job {job.id} ...")
    # try:
    #     job.process()
    # except Exception as err:
    #     logging.info(err)
    # processed.add(job.id)
    # jobs.pop(0)
    # if jobs:
    #     run_jobs()
