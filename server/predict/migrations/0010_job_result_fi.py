# Generated by Django 3.2.6 on 2021-08-20 20:39

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("predict", "0009_job_task_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="result_fi",
            field=models.FilePathField(
                null=True,
                validators=[
                    django.core.validators.FileExtensionValidator(".tsv")
                ],
                verbose_name="Result File Path",
            ),
        ),
    ]