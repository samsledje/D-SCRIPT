from django.db import models

# Create your models here.

class SinglePair(models.Model):
    title = models.CharField(max_length=120)
    protein1 = models.CharField(max_length=60)
    protein2 = models.CharField(max_length=60)
    sequence1 = models.TextField()
    sequence2 = models.TextField()
    probability = models.DecimalField(decimal_places=5, max_digits=6)

    def __str__(self):
        return self.title

class ManyPair(models.Model):
    title = models.CharField(max_length=120)
    pairs = models.FileField(upload_to='pairs/')
    sequences = models.FileField(upload_to='sequences/')
    predictions = models.TextField()
    # predictions = models.FileField(upload_to='predictions/')

    def __str__(self):
        return self.title

class AllPair(models.Model):
    title = models.CharField(max_length=120)
    sequences = models.FileField(upload_to='sequences/')
    predictions = models.TextField()

    def __str__(self):
        return self.title

class PairsUpload(models.Model):
    """
    Model for uploaded pairs file
    """
    pairs = models.FileField(upload_to='pairs/')

class PairsInput(models.Model):
    """
    Model for inputted pairs
    """
    pairs = models.TextField()

class SeqsUpload(models.Model):
    """
    Model for uploaded sequences file
    """
    seqs = models.FileField(upload_to='seqs/')

class SeqsInput(models.Model):
    """
    Model for inputted sequences
    """
    seqs = models.TextField()

class PredictionJob(models.Model):
    """
    Class representing a prediction job
    """
    email = models.TextField()