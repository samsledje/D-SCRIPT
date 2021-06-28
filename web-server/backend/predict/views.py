from django.shortcuts import render
from rest_frameowrk import viewsets
from .serializers import PredictionSerializer
from .models import Prediction

# Create your views here.

class PredictionView(viewsets.ModelViewSet):
    serializer_class = PredictionSerializer
    queryset = Prediction.objects.all()