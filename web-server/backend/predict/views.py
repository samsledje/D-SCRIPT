from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import PredictionSerializer
from .models import Prediction

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
        print (os.getcwd())
        return Response(None)

# class PredictionView(viewsets.ModelViewSet):
#     serializer_class = PredictionSerializer
#     queryset = Prediction.objects.all()