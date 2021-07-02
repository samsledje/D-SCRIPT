from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd

from .serializers import PredictionSerializer, FilePredictionSerializer
from .models import Prediction, FilePrediction

from .api import dscript

import os


# Create your views here.

@api_view(['GET', 'POST'])
def prediction_list(request):
    """
    List all predictions, or create a new prediction.
    """
    if request.method == 'GET':
        predictions = Prediction.objects.all()
        serializer = PredictionSerializer(predictions, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        data = request.data.copy() #dictionary
        data['probability'] = dscript.pair_predict(data['sequence1'], data['sequence2'])
        serializer = PredictionSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('NOT A  POST')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST'])
def file_prediction_list(request):
    """
    List all file predictions, or create a new set of predictions
    """
    if request.method == 'GET':
        file_predictions = FilePrediction.objects.all()
        serializer = FilePredictionSerializer(file_predictions, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        print(request.FILES)
        print(request.data)
        print(request.data['pairs'])
        try:
            pairs = pd.read_csv(request.data['pairs'], sep='\t', header=None)
            all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
            print(all_prots)
        except:
            pass
        return Response(None)

# class PredictionView(viewsets.ModelViewSet):
#     serializer_class = PredictionSerializer
#     queryset = Prediction.objects.all()