import torch
import torch.nn as nn
import torch.nn.functional as F

### Changing to transformer?


class ProteinConv(nn.Module):
    """ """

    def __init__(
        self,
        no_filters,
        no_dims,
        no_channels,
        window_size,
        stride=1,
        dropout_p=0.2,
        activation="tanh",
    ):
        super(ProteinConv, self).__init__()

        activations = {
            "tanh": torch.tanh,
            "sigmoid": F.sigmoid,
            "relu": F.relu,
        }

        # No of filters x No of channels x Window size x No dims
        self.W = nn.Parameter(
            torch.randn(
                no_filters,
                no_channels,
                no_dims,
                window_size,
                dtype=torch.float32,
            )
        )

        self.stride = stride

        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)

        self.dims = no_dims
        self.channels = no_channels
        self.filters = no_filters
        self.window = window_size
        self.activation = activations[activation]

    def attention(self, p1folds, p2folds):
        """
        `p1folds` and `p2folds` should have dimensions N x H1' x 1 x C x D x window
        """
        H1_ = p1folds.size(1)
        H2_ = p2folds.size(1)
        p1folds = torch.sum(
            p1folds * self.W.unsqueeze(0).unsqueeze(0), dim=[3, 5]
        )  # N x H1' x 1 x C x D x window => N x H1' x F x D
        p2folds = torch.sum(
            p2folds * self.W.unsqueeze(0).unsqueeze(0), dim=[3, 5]
        )  # N x H2' x 1 x C x D x window => N x H2' x F x D

        p1folds = torch.transpose(p1folds, 1, 2)  # => N x F x H1' x D
        p2folds = torch.transpose(p2folds, 1, 2)  # => N x F x H2' x D

        att = torch.matmul(
            p1folds, torch.transpose(p2folds, 2, 3)
        ) / torch.sqrt(
            H1_ * H2_
        )  # => N x F x H1' x H2'
        p1folds = torch.matmul(
            F.softmax(att, dim=3), p2folds
        )  # => N x F x H1' x D
        p2folds = torch.matmul(
            F.softmax(torch.transpose(att, 2, 2), dim=3), p1folds
        )  # => N x F x H2' x D

        # Output should be of form => N x (H1'|H2') x F x D
        return torch.transpose(p1folds, 1, 2), torch.transpose(p2folds, 1, 2)

    def forward(self, prot1, prot2):
        """
        `prot1` and `prot2` both represent the protein embeddings of the form `No batch x No squence x No channels x Dim`. We use self.W to transform both `prot1` and `prot2`
        """

        N1, H1, C1, D1 = prot1.shape
        N2, H2, C2, D2 = prot2.shape

        assert (N1, C1, D1) == (
            N2,
            C2,
            D2,
        )  # Only difference between the proteins should be their sequence length

        # unfold prot1 on the third dimension
        p1folds = prot1.unfold(1, self.window, self.stride).unsqueeze(
            2
        )  # N x H1' x 1 x C x D x window
        p2folds = prot2.unfold(1, self.window, self.stride).unsqueeze(
            2
        )  # N x H2' x 1 x C x D x window

        p1folds, p2folds = self.attention(p1folds, p2folds)

        # W.dim == F x C x D x window
        #         p1sum  = torch.sum(self.activation(p1folds * self.W), dim = 1).unsqueeze(1) # N x H1' x F x C x D x window => N x 1 x F x C x D x window
        #         p2sum  = torch.sum(self.activation(p2folds * self.W), dim = 1).unsqueeze(1) # N x H2' x F x C x D x window => N x 1 x F x C x D x window

        #         p1out  = torch.sum(p1folds * p2sum, dim = [3, 5]) # N x H1' x F x C x D1 x window => N x H1' x F x D
        #         p2out  = torch.sum(p2folds * p1sum, dim = [3, 5]) # N x H1' x F x C x D1 x window => N x H1' x F x D
        return self.drop1(self.activation(p1folds)), self.drop2(
            self.activation(p2folds)
        )
