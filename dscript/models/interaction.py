import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-k(x-x_0))}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Parameters:
        - x0: The value of the sigmoid midpoint
        - k: The slope of the sigmoid - trainable
    Examples:
        >>> logAct = LogisticActivation(0, 5)
        >>> x = torch.randn(256)
        >>> x = logAct(x)
    """

    def __init__(self, x0 = 0, k = 1, train=False):
        """
        Initialization
        INPUT:
            - x0: The value of the sigmoid midpoint
            - k: The slope of the sigmoid - trainable
            - train: Whether to make k a trainable parameter
            x0 and k are initialized to 0,1 respectively
            Behaves the same as torch.sigmoid by default
        """
        super(LogisticActivation,self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        """
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1).squeeze()
        return o

    def clip(self):
        self.k.data.clamp_(min=0)

class ModelInteraction(nn.Module):
    def __init__(self, embedding, contact, use_cuda, pool_size=9, theta_init=1, lambda_init = 0, gamma_init = 0, use_W=True):
        super(ModelInteraction, self).__init__()
        self.use_cuda = use_cuda
        self.use_W = use_W
        self.activation = LogisticActivation(x0=0.5, k = 20)

        self.embedding = embedding
        self.contact = contact

        if self.use_W:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.maxPool = nn.MaxPool2d(pool_size,padding=pool_size//2)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

    def clip(self):
        self.contact.clip()

        if self.use_W:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)

    def embed(self, x):
        if self.embedding is None:
            return x
        else:
            return self.embedding(x)

    def cpred(self, z0, z1):
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        B = self.contact.cmap(e0, e1)
        C = self.contact.predict(B)
        return C

    def map_predict(self, z0, z1):

        C = self.cpred(z0, z1)

        if self.use_W:
            # Create contact weighting matrix
            N, M = C.shape[2:]

            x1 = torch.from_numpy(-1 * ((np.arange(N)+1 - ((N+1)/2)) / (-1 * ((N+1)/2)))**2).float()
            if self.use_cuda:
                x1 = x1.cuda()
            x1 = torch.exp(self.lambda_ * x1)

            x2 = torch.from_numpy(-1 * ((np.arange(M)+1 - ((M+1)/2)) / (-1 * ((M+1)/2)))**2).float()
            if self.use_cuda:
                x2 = x2.cuda()
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta

            yhat = C * W

        else:
            yhat = C

        yhat = self.maxPool(yhat)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        phat = self.activation(phat)
        return C, phat

    def predict(self, z0, z1):
        _, phat = self.map_predict(z0,z1)
        return phat

