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


# Create your views here.

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
        result = dscript.predict(data['pairsIndex'], data['seqsIndex'], data['pairs'], data['seqs'])



# class PredictionView(viewsets.ModelViewSet):
#     serializer_class = PredictionSerializer
#     queryset = Prediction.objects.all()