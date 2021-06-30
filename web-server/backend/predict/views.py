from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import PredictionSerializer
from .models import Prediction

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

# class PredictionView(viewsets.ModelViewSet):
#     serializer_class = PredictionSerializer
#     queryset = Prediction.objects.all()