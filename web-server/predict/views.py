from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd

from .commands import dscript

import os
import uuid

# Create your views here.

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
        predict_file = dscript.predict(self.pairsIndex, self.seqsIndex, self.pairs, self.seqs, self.id)
        dscript.email_results(self.email, predict_file, self.id, title=self.title)
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
        jobs.append(job)
        response = {'id': id, 'first': False}
        if len(jobs) == 1:
            response['first'] = True
        return Response(response)

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