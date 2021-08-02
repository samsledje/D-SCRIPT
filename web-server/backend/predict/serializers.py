from rest_framework import serializers
from .models import SinglePair, ManyPair, AllPair
from .models import PairsUpload, PairsInput, SeqsUpload, SeqsInput, PredictionJob

class SinglePairSerializer(serializers.ModelSerializer):
    class Meta:
        model = SinglePair
        fields = ('id', 'title', 'protein1', 'protein2', 'sequence1', 'sequence2', 'probability')

class ManyPairSerializer(serializers.ModelSerializer):
    class Meta:
        model = ManyPair
        fields = ('id', 'title', 'pairs', 'sequences', 'predictions')

class AllPairSerializer(serializers.ModelSerializer):
    class Meta:
        model = AllPair
        fields = ('id', 'title', 'sequences', 'predictions')

class PairsUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = PairsUpload
        fields = ('id', 'pairs')

class PairsInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = PairsInput
        fields = ('id', 'pairs')

class SeqsUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = SeqsUpload
        fields = ('id', 'seqs')

class SeqsInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = SeqsInput
        fields = ('id', 'seqs')

class PredictionJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionJob
        fields = ('id', 'email')
