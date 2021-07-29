from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd

from .serializers import SinglePairSerializer, ManyPairSerializer, AllPairSerializer
from .serializers import PairsUploadSerializer, PairsInputSerializer, SeqsUploadSerializer, SeqsInputSerializer, PredictionJobSerializer
from .models import SinglePair, ManyPair, AllPair
from .models import PairsUpload, PairsInput, SeqsUpload, SeqsInput, PredictionJob

from .api import dscript

import os
import uuid


# Create your views here.

jobs = []

class Job():
    def __init__(self, pairsIndex, seqsIndex, pairs, seqs, email, title, id):
        self.pairsIndex = pairsIndex
        self.seqsIndex = seqsIndex
        self.pairs = pairs
        self.seqs = seqs
        self.email = email
        self.title = title
        self.id = id

    def process(self):
        predict_file = dscript.predict(self.pairsIndex, self.seqsIndex, self.pairs, self.seqs, self.id)
        dscript.email_results(self.email, predict_file, self.id, title=self.title)
        return predict_file
    

@api_view(['POST'])
def test_append(request):
    test.append(1)
    return Response(test)

@api_view(['GET', 'POST'])
def single_pair_predict(request):
    """
    List all single pair predictions, or create a new prediction.
    """
    if request.method == 'GET':
        predictions = SinglePair.objects.all()
        serializer = SinglePairSerializer(predictions, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        data = request.data.copy() #dictionary
        data['probability'] = dscript.single_pair_predict(data['sequence1'], data['sequence2'])
        serializer = SinglePairSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('NOT A VALID POST')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST'])
def many_pair_predict(request):
    """
    List all many pair predictions, or create a new set of predictions
    """
    if request.method == 'GET':
        predictions = ManyPair.objects.all()
        serializer = ManyPairSerializer(predictions, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        data = request.data.copy()
        data['predictions'] = dscript.many_pair_predict(data['title'], data['pairs'], data['sequences'])
        serializer = ManyPairSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('NOT A VALID POST')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST'])
def all_pair_predict(request):
    """
    List all 'all pair' predictions, or create a new set of predictions
    """
    if request.method == 'GET':
        predictions = AllPair.objects.all()
        serializer = AllPairSerializer(predictions, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        data = request.data.copy()
        data['predictions'] = dscript.all_pair_predict(data['title'], data['sequences'])
        serializer = AllPairSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('NOT A VALID POST')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

pairsIndexToSerializer = {'1': PairsUploadSerializer, '2': PairsInputSerializer}
seqsIndexToSerializer = {'1': SeqsUploadSerializer, '2': SeqsInputSerializer}

@api_view(['GET', 'POST'])
def predict(request):
    if request.method == 'POST':
        data = request.data
        if data['pairsIndex'] in ['1', '2']:
            pairsSerializer = pairsIndexToSerializer[data['pairsIndex']](data=data['pairs'])
            if pairsSerializer.is_valid():
                pairsSerializer.save()
            else:
                pass
        seqsSerializer = seqsIndexToSerializer[data['seqsIndex']](data=data['seqs'])
        if seqsSerializer.is_valid():
            seqsSerializer.save()
        else:
            pass
        id = uuid.uuid4()
        job = Job(data['pairsIndex'], data['seqsIndex'], data['pairs'], data['seqs'], data['email'], data['title'], id)
        predict_file = job.process()
        # jobs.append(job)
        # predict_file = dscript.predict(data['pairsIndex'], data['seqsIndex'], data['pairs'], data['seqs'], id)
        # dscript.email_results(data['email'], predict_file, id, title=data['title'])
        return Response(predict_file)



# class PredictionView(viewsets.ModelViewSet):
#     serializer_class = PredictionSerializer
#     queryset = Prediction.objects.all()