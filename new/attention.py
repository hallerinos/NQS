from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call


class SDPA(nn.Module):
    def __init__(self, n_spins, n_hidden, n_layer=1):
        super().__init__()
        self.n_spins = n_spins
        self.n_hidden = n_hidden

        # initialize Q, K & V
        self.Q = nn.Linear(n_hidden, n_hidden, bias=False)
        self.K = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V = nn.Linear(n_hidden, n_hidden, bias=False)
        self.stack = nn.Sequential(
            nn.Linear(n_spins, n_hidden),
            self.attention
        )

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())

    def attention(self, x):
        QKT = self.Q(x) @ self.K(x).T
        attention = torch.nn.Softmax(dim=0)(QKT/4) @ self.V(x)
        return attention


    # returns logprob
    def forward(self, x):
        out = self.stack(x)
        return torch.sum(torch.log(torch.cosh(attention)), dim=-1)