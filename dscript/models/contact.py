# Input: C = NxMxH embedding contact matrix
# Output: S = MxN contact prediction matrix

import torch
import torch.nn as nn
import torch.functional as F

# Choices for f(Z,Z')
class L1(nn.Module):
    # H = 1
    def forward(self, z, zpr):
        return torch.sum(torch.abs(z.unsqueeze(1) - zpr), -1)


class L2(nn.Module):
    # H = 1
    def forward(self, z, zpr):
        return torch.sqrt(torch.sum((z.unsqueeze(1) - zpr) ** 2, -1))


class DotProduct(nn.Module):
    # H = 1
    def forward(self, z, zpr):
        return torch.mm(z, zpr.t())


class OuterProd(nn.Module):
    # H = D
    def forward(self, z, zpr):
        return torch.mul(z.unsqueeze(1), zpr)


class StackAll(nn.Module):
    # H = 3
    def forward(self, z, zpr):
        l1 = L1()
        l2 = L2()
        dp = DotProduct()
        stk = torch.stack((l1.forward(z, zpr), l2.forward(z, zpr), dp.forward(z, zpr)))
        return stk.transpose(0, 2)


class FullyConnected(nn.Module):
    # H = contact_dim
    # c_i,j = Wh where h = [z0_i | z1_j]
    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super(FullyConnected, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation

    def forward(self, z0, z1):
        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)

        c = self.conv(z_cat)
        c = self.activation(c)
        c = self.batchnorm(c)

        return c


class ContactCNN(nn.Module):
    """
    Residue Contact Prediction Module
    """
    def __init__(self, embed_dim, hidden_dim=50, width=7, activation=nn.Sigmoid()):
        super(ContactCNN, self).__init__()

        self.hidden = FullyConnected(embed_dim, hidden_dim)
        # self.hidden = L1(); hidden_dim = 1
        # self.hidden = L2(); hidden_dim = 1
        # self.hidden = DotProduct(); hidden_dim = 1
        # self.hidden = StackAll(); hidden_dim = 3
        # self.hidden = OuterProd(); hidden_dim = embed_dim

        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        self.clip()

    def clip(self):
        # force the conv layer to be transpose invariant
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        C = self.cmap(z0, z1)
        return self.predict(C)

    def cmap(self, z0, z1):
        # z0 is (b,N,D); z1 is (b,M,D)

        # C is (b,N,M,H)
        C = self.hidden(z0, z1)
        return C

    def predict(self, C):

        # S is (b,N,M)
        s = self.conv(C)
        s = self.batchnorm(s)
        s = self.activation(s)
        return s
