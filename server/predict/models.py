import logging

from django.core.validators import EmailValidator, FileExtensionValidator
from django.db import models


class Job(models.Model):
    uuid = models.UUIDField("UUID", primary_key=True)
    title = models.TextField("Title", blank=True)
    email = models.EmailField("Email", validators=[EmailValidator()])

    seq_fi = models.FilePathField(
        "Sequence File Path",
        validators=[FileExtensionValidator(".fasta")],
        null=True,
    )
    pair_fi = models.FilePathField(
        "Pair File Path",
        validators=[FileExtensionValidator(".tsv")],
        null=True,
    )
    n_seqs = models.PositiveIntegerField("Number of Sequences", default=0)
    n_pairs = models.PositiveIntegerField("Number of Pairs", default=0)

    submission_time = models.DateTimeField("Submission Time")
    start_time = models.DateTimeField("Start Time", blank=True, null=True)

    queue_pos = models.IntegerField("Queue Position", default=0)
    n_pairs_done = models.PositiveIntegerField(
        "Number of Pairs Done", blank=True, default=0
    )
    is_running = models.BooleanField("Is Running", default=False)
    is_completed = models.BooleanField("Is Completed", default=False)

    def __str__(self):
        return f"{self.title} ({self.uuid})"
