import torch.nn as nn
import torch.nn.functional as F


class SamplingModel(nn.Module): 
    def __init__(self, sample_size = 100):
        super(SamplingModel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = 2, stride = 2), # N, 1, H, W => N, 5, H', W'
            nn.Tanh(),
            nn.MaxPool2d(3),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels = 5, out_channels = 2, kernel_size = 7, stride = 2), # N, 5, H'', W'' => N, 2, H''', W'''
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Flatten(start_dim = 2),
            nn.LazyLinear(sample_size),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    