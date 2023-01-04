import torch
import torch.nn as nn


class ProteinMaxPool(nn.Module):
    def __init__(self, window, stride=None):
        super(ProteinMaxPool, self).__init__()
        self.window = window
        stride = stride if stride is not None else window
        self.stride = stride

    def forward(self, input):
        N, H, C, D = input.shape
        assert H > self.window
        maxfold, _ = torch.max(
            input.unfold(1, self.window, self.stride), dim=4
        )
        return maxfold
