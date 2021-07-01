from rest_framework import serializers
from .models import Prediction, FilePrediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ('id', 'title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

class FilePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FilePrediction
        fields = ('id', 'title', 'pairs', 'sequences', 'predictions')