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

from .models import Job
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
                status=status.HTTP_501_NOT_IMPLEMENTED,
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


class PredictionServerException(Exception):
    def __init__(
        self, status_code, message="Unspecified PredictionServerException"
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

    def __repr__(self):
        return f"<PredictionServerException:{self.status_code}> {self.message}"


def validate_inputs(seq_path, pair_path):
    try:
        with open(seq_path, "r") as f:
            nam, _ = parse_input(f.read())
        assert len(nam), "You must provide at least one sequence."
        assert (
            len(nam) < settings.DSCRIPT_MAX_SEQS
        ), f"Number of sequences {len(nam)} is larger than the maximum allowed ({settings.DSCRIPT_MAX_SEQS})."
    except AssertionError as err:
        raise PredictionServerException(
            status.HTTP_406_NOT_ACCEPTABLE, f"Sequence parse error: {str(err)}"
        )

    try:
        df = pd.read_csv(pair_path, sep="\t", header=None)
        assert df.shape[1] == 2, "Pairs data frame does not have two columns."
        assert df.shape[0] >= 1, "You must provide at least one pair."
        assert (
            df.shape[0] < settings.DSCRIPT_MAX_PAIRS
        ), f"Number of pairs {df.shape[0]} is larger than the maximum allowed ({settings.DSCRIPT_MAX_PAIRS})."
    except AssertionError as err:
        raise PredictionServerException(
            status.HTTP_406_NOT_ACCEPTABLE, f"Pairs parse error: {str(err)}"
        )

    names_in_pairs = set(df.iloc[:, 0]).union(df.iloc[:, 1])
    names_in_seqs = set(nam)
    if len(names_in_pairs.difference(names_in_seqs)):
        raise PredictionServerException(
            status.HTTP_406_NOT_ACCEPTABLE,
            f"Sequences are requested in the pairs file that are not provided in the sequence file.",
        )

    n_seqs = len(nam)
    n_pairs = len(df)

    return n_seqs, n_pairs


@api_view(["GET", "POST"])
def predict(request):
    """
    Given a prediction input, queues a job for the prediction
    Returns the job id and whether the job comes first to the user
    """
    if request.method == "POST":
        data = request.data
        job_id = uuid.uuid4()

        try:
            seqs_upload = data["seqs"]
            pairs_upload = data["pairs"]

            # Make temporary directory if one does not exist
            os.makedirs(
                f"{tempfile.gettempdir()}/dscript-predictions/", exist_ok=True
            )

            # Write seqs to local file
            seq_path = upload_stream_to_local(
                seqs_upload,
                f"{tempfile.gettempdir()}/dscript-predictions/{job_id}.fasta",
            )

            # Write pairs to local file
            if int(data["pairsIndex"]) == 3:
                pairs_upload = get_all_pairs(seq_path)
            pair_path = upload_stream_to_local(
                pairs_upload,
                f"{tempfile.gettempdir()}/dscript-predictions/{job_id}.tsv",
            )

            # Set result path
            result_path = f"{tempfile.gettempdir()}/dscript-predictions/{job_id}_results.tsv"

            # Validate inputs are properly formatted and allowed
            n_seqs, n_pairs = validate_inputs(seq_path, pair_path)

            logging.debug(n_seqs, seq_path)
            logging.debug(n_pairs, pair_path)

        except PredictionServerException as err:
            logging.debug(err)
            data = {"id": job_id, "submitted": False, "error": err.message}
            return Response(data, status=err.status_code)
        except Exception as err:
            logging.debug(err)
            data = {"id": job_id, "submitted": False, "error": str(err)}
            return Response(data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        job_data = {
            "uuid": job_id,
            "title": data["title"],
            "email": data["email"],
            "seq_fi": seq_path,
            "pair_fi": pair_path,
            "result_fi": result_path,
            "n_seqs": n_seqs,
            "n_pairs": n_pairs,
            "submission_time": datetime.utcnow(),
            "n_pairs_done": 0,
            "is_running": False,
            "is_completed": False,
        }

        # Create job Django item and send off task to start the job
        job_m = Job(**job_data)
        job_m.save()
        job_async = process_job.delay(job_m.uuid)
        async_job_dict[job_id] = job_async

        data = {"id": job_m.uuid, "submitted": True, "error": None}
        return Response(data, status=status.HTTP_200_OK)


@api_view(["GET"])
def get_position(request, uuid):
    logging.info(f" # Getting Position for {uuid} ...")

    if uuid in async_job_dict.keys():
        job_async = async_job_dict[uuid]
        job_state = job_async.state
        if job_state == "SUCCESS" or job_state == "FAILURE":
            _ = job_async.get()
            del async_job_dict[uuid]
    else:
        job = Job.objects.get(pk=uuid)
        job_state = job.task_status

    logging.debug(f"Job {uuid} status {job_state}")
    logging.info("# Sending response")
    return Response({"id": uuid, "status": job_state})
