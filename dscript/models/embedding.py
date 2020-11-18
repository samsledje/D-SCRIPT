"""
Embedding model classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class IdentityEmbed(nn.Module):
    """
    Does not reduce the dimension of the language model embeddings, just passes them through to the contact model.
    """
    def forward(self, x):
        """
        :param x: Input language model embedding :math:`(b \\times N \\times d_0)`
        :type x: torch.Tensor
        :return: Same embedding
        :rtype: torch.Tensor
        """
        return x


class FullyConnectedEmbed(nn.Module):
    """
    Protein Projection Module. Takes embedding from language model and outputs low-dimensional interaction aware projection.

    :param nin: Size of language model output
    :type nin: int
    :param nout: Dimension of projection
    :type nout: int
    :param dropout: Proportion of weights to drop out [default: 0.5]
    :type dropout: float
    :param activation: Activation for linear projection model
    :type activation: torch.nn.Module
    """
    def __init__(self, nin, nout, dropout=0.5, activation=nn.ReLU()):
        super(FullyConnectedEmbed, self).__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout

        self.transform = nn.Linear(nin, nout)
        self.drop = nn.Dropout(p=self.dropout_p)
        self.activation = activation

    def forward(self, x):
        """
        :param x: Input language model embedding :math:`(b \\times N \\times d_0)`
        :type x: torch.Tensor
        :return: Low dimensional projection of embedding
        :rtype: torch.Tensor
        """
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t


class SkipLSTM(nn.Module):
    """
    Language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    Loaded with pre-trained weights in embedding function.

    :param nin: Input dimension of amino acid one-hot [default: 21]
    :type nin: int
    :param nout: Output dimension of final layer [default: 100]
    :type nout: int
    :param hidden_dim: Size of hidden dimension [default: 1024]
    :type hidden_dim: int
    :param num_layers: Number of stacked LSTM models [default: 3]
    :type num_layers: int
    :param dropout: Proportion of weights to drop out [default: 0]
    :type dropout: float
    :param bidirectional: Whether to use biLSTM vs. LSTM
    :type bidirectional: bool
    """
    def __init__(self, nin=21, nout=100, hidden_dim=1024, num_layers=3, dropout=0, bidirectional=True):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
            self.layers.append(f)
            if bidirectional:
                dim = 2 * hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim * num_layers + nin
        if bidirectional:
            n = 2 * hidden_dim * num_layers + nin

        self.proj = nn.Linear(n, nout)

    def to_one_hot(self, x):
        """
        Transform numeric encoded amino acid vector to one-hot encoded vector

        :param x: Input numeric amino acid encoding :math:`(N)`
        :type x: torch.Tensor
        :return: One-hot encoding vector :math:`(N \\times n_{in})`
        :rtype: torch.Tensor
        """
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        """
        :param x: Input numeric amino acid encoding :math:`(N)`
        :type x: torch.Tensor
        :return: Concatenation of all hidden layers :math:`(N \\times (n_{in} + 2 \\times \\text{num_layers} \\times \\text{hidden_dim}))`
        :rtype: torch.Tensor
        """
        one_hot = self.to_one_hot(x)
        hs = [one_hot]  # []
        h_ = one_hot
        for f in self.layers:
            h, _ = f(h_)
            # h = self.dropout(h)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        """
        :meta private:
        """
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h, _ = f(h_)
            # h = self.dropout(h)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1, h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z
