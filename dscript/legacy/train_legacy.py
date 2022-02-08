"""
Train a new model.
"""

import sys
import argparse
import h5py
import datetime
import subprocess as sp
import numpy as np
import pandas as pd
import gzip as gz
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import average_precision_score as average_precision

import dscript
from dscript.utils import PairedDataset, collate_paired_sequences
from dscript.models.embedding import (
    IdentityEmbed,
    FullyConnectedEmbed,
)
from dscript.models.contact import ContactCNN
from dscript.models.interaction import ModelInteraction


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument("--train", help="Training data", required=True)
    data_grp.add_argument("--val", help="Validation data", required=True)
    data_grp.add_argument(
        "--embedding", help="h5 file with embedded sequences", required=True
    )
    data_grp.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        help="Set flag to not augment data by adding (B A) for all pairs (A B)",
    )

    # Embedding model
    proj_grp.add_argument(
        "--projection-dim",
        type=int,
        default=100,
        help="Dimension of embedding projection layer (default: 100)",
    )
    proj_grp.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="Parameter p for embedding dropout layer (default: 0.5)",
    )

    # Contact model
    contact_grp.add_argument(
        "--hidden-dim",
        type=int,
        default=50,
        help="Number of hidden units for comparison layer in contact prediction (default: 50)",
    )
    contact_grp.add_argument(
        "--kernel-width",
        type=int,
        default=7,
        help="Width of convolutional filter for contact prediction (default: 7)",
    )

    # Interaction Model
    inter_grp.add_argument(
        "--no-w",
        action="store_false",
        dest="use_w",
        help="Don't use weight matrix in interaction prediction model",
    )
    inter_grp.add_argument(
        "--pool-width",
        type=int,
        default=9,
        help="Size of max-pool in interaction model (default: 9)",
    )

    # Training
    train_grp.add_argument(
        "--negative-ratio",
        type=int,
        default=10,
        help="Number of negative training samples for each positive training sample (default: 10)",
    )
    train_grp.add_argument(
        "--epoch-scale",
        type=int,
        default=1,
        help="Report heldout performance every this many epochs (default: 1)",
    )
    train_grp.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )
    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Minibatch size (default: 25)",
    )
    train_grp.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="L2 regularization (default: 0)",
    )
    train_grp.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    train_grp.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.35,
        help="Weight on the similarity objective (default: 0.35)",
    )

    # Output
    misc_grp.add_argument(
        "-o", "--outfile", help="Output file path (default: stdout)"
    )
    misc_grp.add_argument(
        "--save-prefix", help="Path prefix for saving models"
    )
    misc_grp.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    misc_grp.add_argument(
        "--checkpoint", help="Checkpoint model to start training from"
    )

    return parser


def predict_interaction(model, n0, n1, tensors, use_cuda):
    """
    Predict whether a list of protein pairs will interact.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool
    """

    b = len(n0)

    p_hat = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()

        p_hat.append(model.predict(z_a, z_b))
    p_hat = torch.stack(p_hat, 0)
    return p_hat


def predict_cmap_interaction(model, n0, n1, tensors, use_cuda):
    """
    Predict whether a list of protein pairs will interact, as well as their contact map.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool
    """

    b = len(n0)

    p_hat = []
    c_map_mag = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()

        cm, ph = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
    p_hat = torch.stack(p_hat, 0)
    c_map_mag = torch.stack(c_map_mag, 0)
    return c_map_mag, p_hat


def interaction_grad(model, n0, n1, y, tensors, use_cuda, weight=0.35):
    """
    Compute gradient and backpropagate loss for a batch.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param y: Interaction labels
    :type y: torch.Tensor
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool
    :param weight: Weight on the contact map magnitude objective. BCE loss is :math:`1 - \\text{weight}`.
    :type weight: float

    :return: (Loss, number correct, mean square error, batch size)
    :rtype: (torch.Tensor, int, torch.Tensor, int)
    """

    c_map_mag, p_hat = predict_cmap_interaction(
        model, n0, n1, tensors, use_cuda
    )
    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
    cmap_loss = torch.mean(c_map_mag)
    loss = (weight * bce_loss) + ((1 - weight) * cmap_loss)
    b = len(p_hat)

    # backprop loss
    loss.backward()

    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()

    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        y = y.float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

    return loss, correct, mse, b


def interaction_eval(model, test_iterator, tensors, use_cuda):
    """
    Evaluate test data set performance.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param test_iterator: Test data iterator
    :type test_iterator: torch.utils.data.DataLoader
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool

    :return: (Loss, number correct, mean square error, precision, recall, F1 Score, AUPR)
    :rtype: (torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    p_hat = []
    true_y = []

    for n0, n1, y in test_iterator:
        p_hat.append(predict_interaction(model, n0, n1, tensors, use_cuda))
        true_y.append(y)

    y = torch.cat(true_y, 0)
    p_hat = torch.cat(p_hat, 0)

    if use_cuda:
        y.cuda()
        p_hat = torch.Tensor([x.cuda() for x in p_hat])
        p_hat.cuda()

    loss = F.binary_cross_entropy(p_hat.float(), y.float()).item()
    b = len(y)

    with torch.no_grad():
        guess_cutoff = torch.Tensor([0.5]).float()
        p_hat = p_hat.float()
        y = y.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

        tp = torch.sum(y * p_hat).item()
        pr = tp / torch.sum(p_hat).item()
        re = tp / torch.sum(y).item()
        f1 = 2 * pr * re / (pr + re)

    y = y.cpu().numpy()
    p_hat = p_hat.data.cpu().numpy()

    aupr = average_precision(y, p_hat)

    return loss, correct, mse, pr, re, f1, aupr


def main(args):
    """
    Run training from arguments.

    :meta private:
    """

    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")

    print(f'# Called as: {" ".join(sys.argv)}', file=output)
    if output is not sys.stdout:
        print(f'Called as: {" ".join(sys.argv)}')

    # Set device
    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(
            f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=output,
        )
    else:
        print("# Using CPU", file=output)
        device = "cpu"

    batch_size = args.batch_size

    train_fi = args.train
    test_fi = args.val
    augment = args.augment
    embedding_h5 = args.embedding
    h5fi = h5py.File(embedding_h5, "r")

    print(f"# Loading training pairs from {train_fi}...", file=output)
    output.flush()

    train_df = pd.read_csv(train_fi, sep="\t", header=None)
    if augment:
        train_n0 = pd.concat((train_df[0], train_df[1]), axis=0).reset_index(
            drop=True
        )
        train_n1 = pd.concat((train_df[1], train_df[0]), axis=0).reset_index(
            drop=True
        )
        train_y = torch.from_numpy(
            pd.concat((train_df[2], train_df[2])).values
        )
    else:
        train_n0, train_n1 = train_df[0], train_df[1]
        train_y = torch.from_numpy(train_df[2].values)

    print(f"# Loading testing pairs from {test_fi}...", file=output)
    output.flush()

    test_df = pd.read_csv(test_fi, sep="\t", header=None)
    test_n0, test_n1 = test_df[0], test_df[1]
    test_y = torch.from_numpy(test_df[2].values)
    output.flush()

    train_pairs = PairedDataset(train_n0, train_n1, train_y)
    pairs_train_iterator = torch.utils.data.DataLoader(
        train_pairs,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )

    test_pairs = PairedDataset(test_n0, test_n1, test_y)
    pairs_test_iterator = torch.utils.data.DataLoader(
        test_pairs,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )

    output.flush()

    print(f"# Loading embeddings", file=output)
    tensors = {}
    all_proteins = (
        set(train_n0)
        .union(set(train_n1))
        .union(set(test_n0))
        .union(set(test_n1))
    )
    for prot_name in tqdm(all_proteins):
        tensors[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])

    use_cuda = (args.device > -1) and torch.cuda.is_available()

    if args.checkpoint is None:

        projection_dim = args.projection_dim
        dropout_p = args.dropout_p
        embedding = FullyConnectedEmbed(
            6165, projection_dim, dropout=dropout_p
        )
        print("# Initializing embedding model with:", file=output)
        print(f"\tprojection_dim: {projection_dim}", file=output)
        print(f"\tdropout_p: {dropout_p}", file=output)

        # Create contact model
        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        print("# Initializing contact model with:", file=output)
        print(f"\thidden_dim: {hidden_dim}", file=output)
        print(f"\tkernel_width: {kernel_width}", file=output)

        contact = ContactCNN(projection_dim, hidden_dim, kernel_width)

        # Create the full model
        use_W = args.use_w
        pool_width = args.pool_width
        print("# Initializing interaction model with:", file=output)
        print(f"\tpool_width: {pool_width}", file=output)
        print(f"\tuse_w: {use_W}", file=output)
        model = ModelInteraction(
            embedding, contact, use_W=use_W, pool_size=pool_width
        )

        print(model, file=output)

    else:
        print(
            "# Loading model from checkpoint {}".format(args.checkpoint),
            file=output,
        )
        model = torch.load(args.checkpoint)
        model.use_cuda = use_cuda

    if use_cuda:
        model = model.cuda()

    # Train the model
    lr = args.lr
    wd = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    report_steps = args.epoch_scale
    inter_weight = args.lambda_
    cmap_weight = 1 - inter_weight
    digits = int(np.floor(np.log10(num_epochs))) + 1
    save_prefix = args.save_prefix
    if save_prefix is None:
        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    print(f'# Using save prefix "{save_prefix}"', file=output)
    print(f"# Training with Adam: lr={lr}, weight_decay={wd}", file=output)
    print(f"\tnum_epochs: {num_epochs}", file=output)
    print(f"\tepoch_scale: {report_steps}", file=output)
    print(f"\tbatch_size: {batch_size}", file=output)
    print(f"\tinteraction weight: {inter_weight}", file=output)
    print(f"\tcontact map weight: {cmap_weight}", file=output)
    output.flush()

    batch_report_fmt = (
        "# [{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}"
    )
    epoch_report_fmt = "# Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"

    N = len(pairs_train_iterator) * batch_size
    for epoch in range(num_epochs):

        model.train()

        n = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        # Train batches
        for (z0, z1, y) in tqdm(
            pairs_train_iterator,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=len(pairs_train_iterator),
        ):

            loss, correct, mse, b = interaction_grad(
                model, z0, z1, y, tensors, use_cuda, weight=inter_weight
            )

            n += b
            delta = b * (loss - loss_accum)
            loss_accum += delta / n

            delta = correct - b * acc_accum
            acc_accum += delta / n

            delta = b * (mse - mse_accum)
            mse_accum += delta / n

            report = (n - b) // 100 < n // 100

            optim.step()
            optim.zero_grad()
            model.clip()

            if report:
                tokens = [
                    epoch + 1,
                    num_epochs,
                    n / N,
                    loss_accum,
                    acc_accum,
                    mse_accum,
                ]
                if output is not sys.stdout:
                    print(batch_report_fmt.format(*tokens), file=output)
                    output.flush()

        if (epoch + 1) % report_steps == 0:
            model.eval()

            with torch.no_grad():

                (
                    inter_loss,
                    inter_correct,
                    inter_mse,
                    inter_pr,
                    inter_re,
                    inter_f1,
                    inter_aupr,
                ) = interaction_eval(
                    model, pairs_test_iterator, tensors, use_cuda
                )
                tokens = [
                    epoch + 1,
                    num_epochs,
                    inter_loss,
                    inter_correct / (len(pairs_test_iterator) * batch_size),
                    inter_mse,
                    inter_pr,
                    inter_re,
                    inter_f1,
                    inter_aupr,
                ]
                print(epoch_report_fmt.format(*tokens), file=output)
                output.flush()

            # Save the model
            if save_prefix is not None:
                save_path = (
                    save_prefix
                    + "_epoch"
                    + str(epoch + 1).zfill(digits)
                    + ".sav"
                )
                print(f"# Saving model to {save_path}", file=output)
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

        output.flush()

    if save_prefix is not None:
        save_path = save_prefix + "_final.sav"
        print(f"# Saving final model to {save_path}", file=output)
        model.cpu()
        torch.save(model, save_path)
        if use_cuda:
            model.cuda()

    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
