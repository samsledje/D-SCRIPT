from django.db import models

# Create your models here.


class Job(models.Model):
    uuid = models.UUIDField()
    title = models.TextField(blank=True)
    email = models.EmailField()
    seqsIndex = models.CharField(max_length=1)
    pairsIndex = models.CharField(max_length=1)
    seqs = models.TextField()
    pairs = models.TextField(blank=True)
    completed = models.BooleanField()

    def __str__(self):
        return self.uuid
