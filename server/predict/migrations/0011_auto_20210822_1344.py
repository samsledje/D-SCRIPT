# Generated by Django 3.2.4 on 2021-08-22 13:44

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("predict", "0010_job_result_fi"),
    ]

    operations = [
        migrations.AlterField(
            model_name="job",
            name="pair_fi",
            field=models.FilePathField(
                null=True,
                path="/var/folders/nq/q5ck0t0n3tnb9l5xd88gz4z40000gp/T/dscript-predictions/",
                validators=[
                    django.core.validators.FileExtensionValidator(".tsv")
                ],
                verbose_name="Pair File Path",
            ),
        ),
        migrations.AlterField(
            model_name="job",
            name="result_fi",
            field=models.FilePathField(
                null=True,
                path="/var/folders/nq/q5ck0t0n3tnb9l5xd88gz4z40000gp/T/dscript-predictions/",
                validators=[
                    django.core.validators.FileExtensionValidator(".tsv")
                ],
                verbose_name="Result File Path",
            ),
        ),
        migrations.AlterField(
            model_name="job",
            name="seq_fi",
            field=models.FilePathField(
                null=True,
                path="/var/folders/nq/q5ck0t0n3tnb9l5xd88gz4z40000gp/T/dscript-predictions/",
                validators=[
                    django.core.validators.FileExtensionValidator(".fasta")
                ],
                verbose_name="Sequence File Path",
            ),
        ),
    ]
