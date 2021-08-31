import sys

sys.path.append("../")

import logging
import os
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import torch
from django.conf import settings
from dotenv import load_dotenv

from dscript.fasta import parse
from dscript.language_model import lm_embed
from dscript.pretrained import get_pretrained

from ..models import Job

load_dotenv()

if settings.DSCRIPT_DEPLOY_ENV:
    outgoing_mail_server = "outgoing.csail.mit.edu"
    outgoing_mail_port = 25
else:
    outgoing_mail_server = "smtp.gmail.com"
    outgoing_mail_port = 465


def predict_pairs(
    uuid,
    device=settings.DSCRIPT_DEVICE,
    model_version=settings.DSCRIPT_MODEL_VERSION,
    **kwargs,
):
    """
    Given specified candidate pairs and protein sequences,
    Creates a .tsv file of interaction predictions and returns the url
    within the temporary directory
    """

    job = Job.objects.get(pk=uuid)

    n_complete = job.n_pairs_done
    seq_file = job.seq_fi
    pair_file = job.pair_fi
    result_file = job.result_fi

    if os.path.exists(result_file):
        if n_complete == job.n_pairs:
            logging.warning(
                f"Job {uuid} started with pairs_done {n_complete} == total_pairs {job.n_pairs}"
            )
        mode = "a"
    else:
        mode = "w+"

    # Set Device
    logging.info("# Setting Device...")
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        logging.info(
            f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
        )
    else:
        logging.info("# Using CPU")

    # Load Model
    logging.info("# Loading Model...")
    try:
        model = get_pretrained(model_version)
        if use_cuda:
            model = model.cuda()
        else:
            model = model.cpu()
            model.use_cuda = False
    except ValueError:
        logging.warning(f"# Model {model_version} not available")
        return

    # Load Sequences
    logging.info("# Loading Sequences...")
    with open(seq_file, "r") as f:
        names, sequences = parse(f)
    seqDict = {n.split()[0]: s for n, s in zip(names, sequences)}
    logging.info(seqDict)

    # Load Pairs
    logging.info("# Loading Pairs...")
    pairs_array = pd.read_csv(pair_file, sep="\t", header=None)
    all_prots = set(pairs_array.iloc[:, 0]).union(pairs_array.iloc[:, 1])

    # Generate Embeddings
    # logging.info("# Generating Embeddings...")
    # embeddings = {}
    # for n in all_prots:
    #     embeddings[n] = lm_embed(seqDict[n], use_cuda)

    # Make Predictions
    logging.info("# Making Predictions...")
    model = model.eval()
    with open(result_file, mode) as f:
        with torch.no_grad():
            for _, (n0, n1) in pairs_array.iloc[n_complete:, :2].iterrows():
                n0 = str(n0)
                n1 = str(n1)
                if n_complete % 50 == 0:
                    job.n_pairs_done = n_complete
                    job.save()
                    f.flush()
                n_complete += 1
                # p0 = embeddings[n0]
                # p1 = embeddings[n1]
                p0 = lm_embed(seqDict[n0], use_cuda)
                p1 = lm_embed(seqDict[n1], use_cuda)
                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()
                try:
                    p = model.predict(p0, p1).item()
                    f.write(f"{n0}\t{n1}\t{p}\n")
                except RuntimeError as e:
                    logging.error(f"{n0} x {n1} skipped - Out of Memory")

    return result_file


def create_message(
    sender_email,
    receiver_email,
    subject,
    body,
    uuid,
    filename=None,
):
    """
    Creates an email message with the appropriate headers
    Returns the message converted to a string
    """
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    if filename:

        # Open PDF file in binary mode
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {uuid}.tsv",
        )

        # Add attachment to message
        message.attach(part)

    # Convert message to string
    text = message.as_string()

    return text


def send_message(sender_email, receiver_email, text):
    # Log in to server using secure context and send email
    if settings.DSCRIPT_DEPLOY_ENV:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        with smtplib.SMTP(outgoing_mail_server, outgoing_mail_port) as server:
            server.starttls(context=context)
            server.sendmail(sender_email, receiver_email, text)
    else:
        context = ssl.create_default_context()
        password = os.getenv("EMAIL_PWD")
        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, context=context
        ) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)


def email_results(
    uuid,
    sender_email=settings.DSCRIPT_SENDER_EMAIL,
):
    """
    Given a user email, target path for prediction file, and job id
    Emails the user the results of their job
    """

    job = Job.objects.get(pk=uuid)

    title = job.title
    receiver_email = job.email
    filename = job.result_fi

    logging.info("# Emailing Results ...")
    if not title:
        subject = f"D-SCRIPT Results for {uuid}"
    else:
        subject = f"D-SCRIPT Results for {title} ({uuid})"
    body = f"These are the results of your D-SCRIPT prediction on job {uuid}"

    text = create_message(
        sender_email, receiver_email, subject, body, uuid, filename
    )

    send_message(sender_email, receiver_email, text)


def email_confirmation(
    uuid,
    sender_email=settings.DSCRIPT_SENDER_EMAIL,
):
    """
    Given a user email, job id, and potential title,
    Emails the user confirming that their job has been submitted.
    """

    job = Job.objects.get(pk=uuid)

    title = job.title
    receiver_email = job.email

    logging.info("# Emailing Confirmation ...")
    if not title:
        subject = f"D-SCRIPT Job {uuid} Submission"
    else:
        subject = f"D-SCRIPT Job {title} ({uuid}) Submission"
    body = f"You have successfully submitted a job with id {uuid} for D-SCRIPT prediction. Keep track of this id to monitor your job status."

    text = create_message(sender_email, receiver_email, subject, body, uuid)

    send_message(sender_email, receiver_email, text)
