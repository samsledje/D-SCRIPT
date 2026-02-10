import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .contact import ContactCNN
from .embedding import FullyConnectedEmbed

from dataclasses import dataclass


@dataclass
class InteractionInputs:
    z0: torch.Tensor
    z1: torch.Tensor

    f0: torch.Tensor = None
    f1: torch.Tensor = None

    b0: torch.Tensor = None
    b1: torch.Tensor = None

    embed_foldseek: bool = False
    embed_backbone: bool = False

    def __post_init__(self):
        if self.embed_foldseek:
            assert self.f0 is not None and self.f1 is not None
            assert isinstance(self.f0, torch.Tensor) and isinstance(
                self.f1, torch.Tensor
            )
            assert (
                self.z0.get_device() == self.f0.get_device()
                and self.z0.get_device() == self.f1.get_device()
            )
            assert (
                self.f0.shape[1] == self.z0.shape[1]
                and self.f1.shape[1] == self.z1.shape[1]
            )
        if self.embed_backbone:
            assert self.b0 is not None and self.b1 is not None
            assert isinstance(self.b0, torch.Tensor) and isinstance(
                self.b1, torch.Tensor
            )
            assert (
                self.z0.get_device() == self.b0.get_device()
                and self.z0.get_device() == self.b1.get_device()
            )
            assert (
                self.b0.shape[1] == self.z0.shape[1]
                and self.b1.shape[1] == self.z1.shape[1]
            )


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
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise

        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.

        :meta private:
        """
        self.k.data.clamp_(min=0)


class PairClassifier2D(nn.Module):
    def __init__(self, hidden=128, p_drop=0.2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, yhat_fused,gamma):
        # compute Q from yhat_fused
        mu = yhat_fused.mean(dim=(1,2,3), keepdim=True)   # [B,1,1,1]
        sigma = yhat_fused.var(dim=(1,2,3), keepdim=True, unbiased=False).clamp_min(1e-6)
        std = sigma.sqrt()
        # [B,1,1,1]
        D = yhat_fused - mu - (gamma * std) 
        temp = 5.0# [B,1,N,M]
        Q = torch.sigmoid(D * temp)                                  # [B,1,N,M]
        phat = (Q * yhat_fused).sum(dim=(1,2,3)) / (Q.sum(dim=(1,2,3)) + 1e-6)
        
        scale = torch.sigmoid(phat).view(-1,1,1,1)      # (0,1)
        inp = yhat_fused * (1.0 + scale)                 
        
        x = self.feat(inp)  # [B,H,N,M]

        # global pooling -> fixed size regardless of N,M
        x_mean = x.mean(dim=(2, 3))  # [B,H]
        x_max = x.amax(dim=(2, 3))  # [B,H]
        x = torch.cat([x_mean, x_max], dim=1)  # [B,2H]

        # default: no augmentation
        x_aug = x 
        lam = None
        index = None

        if self.training:
            B = x.size(0)
            lam = torch.empty(B, 1, device=x.device).uniform_(0.75, 0.95)
            index = torch.randperm(B, device=x.device)
            x_aug = lam * x + (1.0 - lam) * x[index]  # [B,2H]

        x_aug = self.dropout(x_aug)
        logit = self.head(x_aug).squeeze(1)  # [B]

        return logit, x_aug, lam, index


class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        contact,
        use_cuda,
        do_w=True,
        # language_mod_size=25,
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
        super().__init__()
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
        self.pool_size = pool_size
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

        self.xx = nn.Parameter(torch.arange(2000), requires_grad=False)
        ## added aug
        k = 8
        ## need to adjust after set the dims of foldseek and backbone embedding
        D = self.embedding.nout  # = 100
        self.g_proj = nn.Linear(D, k)
        self.yhat_fuse = nn.Conv2d(1 + 2 * k, 1, kernel_size=1, bias=True)
        self.clf = PairClassifier2D(hidden=128, p_drop=0.2)

    def clip(self):
        """
        Clamp model values

        :meta private:
        """
        self.contact.clip()

        with torch.no_grad():
            if self.do_w:
                self.theta.clamp_(0, 1)
                self.lambda_.clamp_(min=0)

            self.gamma.clamp_(min=0)

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

    def cpred(self, inputs):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """
        e0 = self.embed(inputs.z0)
        e1 = self.embed(inputs.z1)

        if inputs.embed_foldseek:
            e0 = torch.concat([e0, inputs.f0], dim=2)
            e1 = torch.concat([e1, inputs.f1], dim=2)

        if inputs.embed_backbone:
            e0 = torch.concat([e0, inputs.b0], dim=2)
            e1 = torch.concat([e1, inputs.b1], dim=2)

        B = self.contact.cmap(e0, e1)
        C = self.contact.predict(B)
        ###added augment
        # print("e0 shape:", e0.shape)
        # print("e1 shape:", e1.shape)
        p0 = e0.mean(dim=1)
        p1 = e1.mean(dim=1)
        int0 = p0 + p1
        int1 = p0 * p1
        # print("int0, int1 shape:", int0.shape, int1.shape)
        ### added return int0, int1
        return C, int0, int1

    # TODO: Temporaru overload to allow downstream (post train/evaluate) methods to work.
    def _build_interaction_inputs(
        self,
        z0,
        z1,
        ### Foldseek embedding added
        embed_foldseek=False,
        f0=None,
        f1=None,
        ### Backbone embedding added
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted contact map, predicted probability of interaction :math:`(b \\times N \\times d_0), (1)`
        :rtype: torch.Tensor, torch.Tensor
        """
        return InteractionInputs(
            z0,
            z1,
            embed_foldseek=embed_foldseek,
            f0=f0,
            f1=f1,
            embed_backbone=embed_backbone,
            b0=b0,
            b1=b1,
        )

    def make_mixup_params(self, batch_size: int, alpha: float, device):
        perm = torch.randperm(batch_size, device=device)

        # Always return lam tensor (never None)
        if alpha is None or alpha <= 0 or batch_size <= 1:
            lam = torch.ones(batch_size, device=device)
        else:
            lam = (
                torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
            )

        return perm, lam

    def map_predict(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], InteractionInputs):
            cpredInputs = args[0]

        if len(args) >= 2:
            cpredInputs = self._build_interaction_inputs(*args, **kwargs)

        C, g_add, g_mul = self.cpred(cpredInputs)

        if self.training and not hasattr(self, "_printed_batch_B"):
            self._printed_batch_B = True
            print("[DEBUG map_predict] C shape =", tuple(C.shape))

        # Ensure g_add/g_mul are [B,D]
        if g_add.ndim == 3:  # [B,1,D]
            g_add = g_add.squeeze(1)
            g_mul = g_mul.squeeze(1)

        if self.do_w:
            N, M = C.shape[2:]
            device = C.device

            xx_N = torch.arange(N, device=device, dtype=C.dtype)
            xx_M = torch.arange(M, device=device, dtype=C.dtype)

            x1 = -1 * torch.square((xx_N + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2)))

            x2 = -1 * torch.square((xx_M + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2)))

            x1 = torch.exp(self.lambda_ * x1)
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta

            yhat = C * W

        else:
            yhat = C

        # ---- fuse global interaction into map (BEFORE pooling is usually better)
        B, _, N, M = yhat.shape

        ga = self.g_proj(g_add)  # [B,k]
        gm = self.g_proj(g_mul)  # [B,k]

        ga_map = ga[:, :, None, None].expand(B, ga.shape[1], N, M)  # [B,k,N,M]
        gm_map = gm[:, :, None, None].expand(B, gm.shape[1], N, M)  # [B,k,N,M]

        yhat_cat = torch.cat([yhat, ga_map, gm_map], dim=1)  # [B,1+2k,N,M]
        yhat_fused = self.yhat_fuse(yhat_cat)  # [B,1,N,M]
        
        if self.do_pool:
            yhat_fused = self.maxPool(yhat_fused)
     
        logit, z, lam, index = self.clf(yhat_fused,self.gamma)  # [B]

        return C, logit, z,lam, index

    # INTERNAL
    def predict(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted probability of interaction
        :rtype: torch.Tensor, torch.Tensor
        """
        _, phat, _,_, _ = self.map_predict(
            z0,
            z1,
            embed_foldseek=embed_foldseek,
            f0=f0,
            f1=f1,
            embed_backbone=embed_backbone,
            b0=b0,
            b1=b1,
        )
        return phat

    def forward(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        """
        :meta private:
        """
        return self.predict(
            z0,
            z1,
            embed_foldseek=embed_foldseek,
            f0=f0,
            f1=f1,
            embed_backbone=embed_backbone,
            b0=b0,
            b1=b1,
        )


class DSCRIPTModel(ModelInteraction, PyTorchModelHubMixin):
    def __init__(
        self,
        emb_nin,
        emb_nout,
        emb_dropout,
        con_embed_dim,
        con_hidden_dim,
        con_width,
        use_cuda,
        emb_activation=nn.ReLU(),
        con_activation=nn.Sigmoid(),
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        embedding = FullyConnectedEmbed(emb_nin, emb_nout, emb_dropout, emb_activation)
        contact = ContactCNN(con_embed_dim, con_hidden_dim, con_width, con_activation)
        super().__init__(
            embedding=embedding,
            contact=contact,
            use_cuda=use_cuda,
            do_w=do_w,
            do_sigmoid=do_sigmoid,
            do_pool=do_pool,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )
