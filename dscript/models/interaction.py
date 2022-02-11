import numpy as np
import torch
import torch.functional as F
import torch.nn as nn


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:

    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`

    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise

        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.

        :meta private:
        """
        self.k.data.clamp_(min=0)


class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        contact,
        use_cuda,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        """
        Main D-SCRIPT model. Contains an embedding and contact model and offers access to those models. Computes pooling operations on contact map to generate interaction probability.

        :param embedding: Embedding model
        :type embedding: dscript.models.embedding.FullyConnectedEmbed
        :param contact: Contact model
        :type contact: dscript.models.contact.ContactCNN
        :param use_cuda: Whether the model should be run on GPU
        :type use_cuda: bool
        :param do_w: whether to use the weighting matrix [default: True]
        :type do_w: bool
        :param do_sigmoid: whether to use a final sigmoid activation [default: True]
        :type do_sigmoid: bool
        :param do_pool: whether to do a local max-pool prior to the global pool
        :type do_pool: bool
        :param pool_size: width of max-pool [default 9]
        :type pool_size: bool
        :param theta_init: initialization value of :math:`\\theta` for weight matrix [default: 1]
        :type theta_init: float
        :param lambda_init: initialization value of :math:`\\lambda` for weight matrix [default: 0]
        :type lambda_init: float
        :param gamma_init: initialization value of :math:`\\gamma` for global pooling [default: 0]
        :type gamma_init: float

        """
        super(ModelInteraction, self).__init__()
        self.use_cuda = use_cuda
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
        if do_sigmoid:
            self.activation = LogisticActivation(x0=0.5, k=20)

        self.embedding = embedding
        self.contact = contact

        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.do_pool = do_pool
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

    def clip(self):
        """
        Clamp model values

        :meta private:
        """
        self.contact.clip()

        if self.do_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)

    def embed(self, x):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z: torch.Tensor
        :return: D-SCRIPT projection :math:`(b \\times N \\times d)`
        :rtype: torch.Tensor
        """
        if self.embedding is None:
            return x
        else:
            return self.embedding(x)

    def cpred(self, z0, z1):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        B = self.contact.cmap(e0, e1)
        C = self.contact.predict(B)
        return C

    def map_predict(self, z0, z1):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted contact map, predicted probability of interaction :math:`(b \\times N \\times d_0), (1)`
        :rtype: torch.Tensor, torch.Tensor
        """

        C = self.cpred(z0, z1)

        if self.do_w:
            # Create contact weighting matrix
            N, M = C.shape[2:]

            x1 = torch.from_numpy(
                -1
                * ((np.arange(N) + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2)))
                ** 2
            ).float()
            if self.use_cuda:
                x1 = x1.cuda()
            # x1 = torch.exp(self.lambda1 * x1)
            x1 = torch.exp(self.lambda_ * x1)

            x2 = torch.from_numpy(
                -1
                * ((np.arange(M) + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2)))
                ** 2
            ).float()
            if self.use_cuda:
                x2 = x2.cuda()
            # x2 = torch.exp(self.lambda2 * x2)
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta

            yhat = C * W

        else:
            yhat = C

        if self.do_pool:
            yhat = self.maxPool(yhat)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        # Q = torch.relu(yhat - mu)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        if self.do_sigmoid:
            phat = self.activation(phat)
        return C, phat

    def predict(self, z0, z1):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted probability of interaction
        :rtype: torch.Tensor, torch.Tensor
        """
        _, phat = self.map_predict(z0, z1)
        return phat

    def forward(self, z0, z1):
        """
        :meta private:
        """
        return self.predict(z0, z1)
