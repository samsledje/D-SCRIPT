import sys

sys.path.append("../")

import logging
import os
import smtplib
import ssl
import tempfile
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from dscript.fasta import parse_input
from dscript.language_model import lm_embed

from ..models import Job

load_dotenv()


def predict_pairs(
    uuid,
    seq_file,
    pair_file,
    device=-1,
    modelPath="dscript-models/human_v1.sav",
    **kwargs,
):
    """
    Given specified candidate pairs and protein sequences,
    Creates a .tsv file of interaction predictions and returns the url
    within the temporary directory
    """

    job = Job.objects.get(pk=uuid)

    # Set Outpath
    os.makedirs(f"{tempfile.gettempdir()}/dscript-predictions/", exist_ok=True)
    out_file = (
        f"{tempfile.gettempdir()}/dscript-predictions/{uuid}_results.tsv"
    )

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
        if use_cuda:
            model = torch.load(modelPath).cuda()
        else:
            model = torch.load(modelPath).cpu()
            model.use_cuda = False
    except FileNotFoundError:
        logging.info(f"# Model {modelPath} not found")
        return

    # Load Sequences
    logging.info("# Loading Sequences...")
    with open(seq_file, "r") as f:
        names, sequences = parse_input(f.read())
    seqDict = {n: s for n, s in zip(names, sequences)}
    logging.info(seqDict)

    # Load Pairs
    logging.info("# Loading Pairs...")
    pairs_array = pd.read_csv(pair_file, sep="\t", header=None)
    all_prots = set(pairs_array.iloc[:, 0]).union(set(pairs_array.iloc[:, 1]))

    # Generate Embeddings
    logging.info("# Generating Embeddings...")
    embeddings = {}
    for n in all_prots:
        embeddings[n] = lm_embed(seqDict[n], use_cuda)

    # Make Predictions
    logging.info("# Making Predictions...")
    n = 0
    model = model.eval()
    with open(out_file, "w+") as f:
        with torch.no_grad():
            for _, (n0, n1) in pairs_array.iloc[:, :2].iterrows():
                n0 = str(n0)
                n1 = str(n1)
                if n % 50 == 0:
                    job.n_pairs_done = n
                    job.save()
                    f.flush()
                n += 1
                p0 = embeddings[n0]
                p1 = embeddings[n1]
                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()
                try:
                    p = model.predict(p0, p1).item()
                    f.write(f"{n0}\t{n1}\t{p}\n")
                except RuntimeError as e:
                    logging.error(f"{n0} x {n1} skipped - Out of Memory")

    return out_file


def email_results(
    receiver_email,
    filename,
    id,
    title=None,
    sender_email="dscript.results@gmail.com",
):
    """
    Given a user email, target path for prediction file, and job id
    Emails the user the results of their job
    """
    logging.info("# Emailing Results ...")
    if not title:
        subject = f"D-SCRIPT Results for {id}"
    else:
        subject = f"D-SCRIPT Results for {title} ({id})"
    body = f"These are the results of your D-SCRIPT prediction on job {id}"
    password = os.getenv("EMAIL_PWD")

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # filename

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
        f"attachment; filename= {id}.tsv",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)
