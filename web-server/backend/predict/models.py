from django.db import models

# Create your models here.

class Prediction(models.Model):
    title = models.CharField(max_length=120)
    protein1 = models.CharField(max_length=60)
    protein2 = models.CharField(max_length=60)
    sequence1 = models.TextField()
    sequence2 = models.TextField()
    probability = models.DecimalField(decimal_places=5, max_digits=6)

    def __str__(self):
        return self.title