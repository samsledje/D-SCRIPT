from rest_framework import serializers
from .models import SinglePair, ManyPair

class SinglePairSerializer(serializers.ModelSerializer):
    class Meta:
        model = SinglePair
        fields = ('id', 'title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

class ManyPairSerializer(serializers.ModelSerializer):
    class Meta:
        model = ManyPair
        fields = ('id', 'title', 'pairs', 'sequences', 'predictions')