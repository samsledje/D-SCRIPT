import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinConcat(nn.Module):
    def __init__(
        self,
        no_dims,
        no_channels,
        window_size,
        op_size,
        stride=1,
        dropout_p=0.2,
        activation="tanh",
    ):
        super(ProteinConcat, self).__init__()

        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": F.sigmoid,
        }

        self.drop = nn.Dropout(p=dropout_p)
        self.W = nn.Parameter(
            torch.randn(
                1, 1, no_channels, no_dims, window_size, dtype=torch.float32
            )
        )
        self.lin = nn.Linear(no_dims * window_size * 2, op_size)
        self.window_size = window_size
        self.stride = stride
        self.activation = activations[activation]

    def forward(self, p1, p2):
        # p1, p2 = N x (H1, H2) x C x D

        p1stride = p1.unfold(
            1, self.window_size, self.stride
        )  # N x H1' x C x D x W
        p2stride = p2.unfold(
            1, self.window_size, self.stride
        )  # N x H2' x C x D x W

        p1sum = torch.sum(
            self.activation(p1stride * self.W), dim=[1, 2]
        )  # N x D x W
        p2sum = torch.sum(
            self.activation(p2stride * self.W), dim=[1, 2]
        )  # N x D x W

        pout = torch.flatten(
            torch.cat([p1sum + p2sum, p1sum * p2sum], dim=2), start_dim=1
        )  # N x (2 x D x W)

        return self.drop(self.lin(self.activation(pout)))
