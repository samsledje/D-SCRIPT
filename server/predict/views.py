from django.views.generic import View
from django.conf import settings
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd

from .api import dscript

from .serializers import JobSerializer

import os
import uuid
import logging

# Create your views here.

class FrontendAppView(View):
    """
    Serves the compiled frontend entry point (only works if you have run `npm
    run build`).
    """

    def get(self, request):
            logging.info(os.path.join(settings.REACT_APP_DIR, 'build', 'index.html'))
            try:
                with open(os.path.join(settings.REACT_APP_DIR, 'build', 'index.html')) as f:
                    return HttpResponse(f.read())
            except FileNotFoundError:
                logging.exception('Production build of app not found')
                return HttpResponse(
                    """
                    This URL is only used when you have built the production
                    version of the app. Visit http://localhost:3000/ instead, or
                    run `yarn run build` to test the production version.
                    """,
                    status=501,
                )

jobs = []

processed = set()

class Job():
    def __init__(self, pairsIndex, seqsIndex, pairs, seqs, email, title, id):
        self.pairsIndex = pairsIndex
        self.seqsIndex = seqsIndex
        self.pairs = pairs
        self.seqs = seqs
        self.email = email
        self.title = title
        self.id = id
        if pairsIndex == '1':
            self.pairs = ''
            for line in pairs:
                self.pairs += line.decode('utf-8').replace('\r', '').replace('\t', ',')
        if seqsIndex == '1':
            self.seqs = ''
            for line in seqs:
                self.seqs += line.decode('utf-8')
        print(self.seqs)

    def process(self):
        predict_file = dscript.predict(self.seqs, self.pairsIndex, self.pairs, self.id)
        try:
            dscript.email_results(self.email, predict_file, self.id, title=self.title)
        except:
            print('Not a valid email')
        return predict_file

@api_view(['GET', 'POST'])
def predict(request):
    """
    Given a prediction input, queues a job for the prediction
    Returns the job id and whether the job comes first to the user
    """
    if request.method == 'POST':
        data = request.data
        id = uuid.uuid4()
        job = Job(data['pairsIndex'], data['seqsIndex'], data['pairs'], data['seqs'], data['email'], data['title'], id)
        job_data = {'uuid': job.id, 'title': job.title, 'email': job.email, 'seqsIndex': job.seqsIndex, 'pairsIndex': job.pairsIndex, 'seqs': job.seqs, 'pairs': job.pairs, 'completed': False}
        serializer = JobSerializer(data=job_data)
        if serializer.is_valid():
            serializer.save()
            jobs.append(job)
            response = {'id': id, 'first': False}
            if len(jobs) == 1:
                response['first'] = True
            return Response(response)
        else:
            print(serializer.errors)
            print('NOT A VALID SERIALIZER')

@api_view(['GET'])
def get_pos(request, id):
    print(f' # Getting Queue Position for {id} ...')
    if id in processed:
        return Response({'position': -1, 'inQueue': False})
    for i in range(len(jobs)):
        if jobs[i].id == id:
            return Response({'position': i+1, 'inQueue': True})
    return Response({'position': 0, 'inQueue': False})


@api_view(['POST'])
def process_jobs(request):
    run_jobs()
    return Response({})

def run_jobs():
    job = jobs[0]
    print(f' # Processing Job {job.id} ...')
    try:
        job.process()
    except:
        pass
    processed.add(job.id)
    jobs.pop(0)
    if jobs:
        run_jobs()