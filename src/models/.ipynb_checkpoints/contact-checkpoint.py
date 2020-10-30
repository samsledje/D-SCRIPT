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
        return torch.sqrt(torch.sum((z.unsqueeze(1) - zpr)**2, -1))

class FullyConnected(nn.Module):
    # H = contact_dim
    # c_i,j = Wh where h = [z0_i | z1_j]
    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super(FullyConnected, self).__init__()
        
        self.D = embed_dim
        self.H = hidden_dim
        
        self.conv = nn.Conv2d(2*self.D, self.H, 1)
        torch.nn.init.normal_(self.conv.weight)
        torch.nn.init.uniform_(self.conv.bias, 0, 0)
        
        self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation
        
    def forward(self, z0, z1):
        z0 = z0.transpose(1,2)
        z1 = z1.transpose(1,2)
        
        z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)
        
        c = self.conv(z_cat)
        c = self.activation(c)
        c = self.batchnorm(c)
        
        return c
    
# Contact Prediction Model
class ContactCNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim=50, width=7, output_dim=1, activation=nn.Sigmoid()):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.hidden = FullyConnected(self.embed_dim, self.hidden_dim)
        
        self.conv = nn.Conv2d(self.hidden_dim, self.output_dim, width, padding=width//2)
        torch.nn.init.normal_(self.conv.weight)
        torch.nn.init.uniform_(self.conv.bias, 0, 0)
        
        self.batchnorm = nn.BatchNorm2d(self.output_dim)
        self.activation = activation
        self.clip()

    def clip(self):
        # force the conv layer to be transpose invariant
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5*(w + w.transpose(2,3))

    def forward(self, z0, z1):
        B = self.broadcast(z0, z1)
        C = self.predict(B)
        return C

    def broadcast(self, z0, z1):
        B = self.hidden(z0, z1)
        return B
    
    def predict(self, B):
        C = self.conv(B)
        C = self.batchnorm(C)
        C = self.activation(C)
        return C
    
class ContactCNN_v2(nn.Module):
    def __init__(self, embed_dim, hidden_dim=50, width=7, output_dim=1, activation=nn.Sigmoid()):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.hidden = FullyConnected(self.embed_dim, self.hidden_dim)
        
        self.conv = nn.Conv2d(self.hidden_dim, self.output_dim, width, padding=width//2)
        torch.nn.init.normal_(self.conv.weight)
        torch.nn.init.uniform_(self.conv.bias, 0, 0)
        
        self.batchnorm = nn.BatchNorm2d(self.output_dim)
        self.activation = activation
        self.clip()

    def clip(self):
        # force the conv layer to be transpose invariant
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5*(w + w.transpose(2,3))

    def forward(self, z0, z1):
        B = self.broadcast(z0, z1)
        C = self.predict(B)
        return C

    def broadcast(self, z0, z1):
        B = self.hidden(z0, z1)
        return B
    
    def predict(self, B):
        C = self.conv(B)
        C = self.batchnorm(C)
        C = self.activation(C)
        return C