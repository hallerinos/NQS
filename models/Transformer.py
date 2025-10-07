from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call


class SDPA(nn.Module):
    def __init__(self, n_spins, n_hidden, n_batch=4):
        super().__init__()
        self.n_spins = n_spins
        self.n_hidden = n_hidden

        self.stack = nn.Sequential(
            nn.Linear(n_spins, n_spins),
            attention(n_spins, n_batch),
            nn.Linear(n_hidden, n_hidden),
        )

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        out = self.stack(x)
        return torch.sum(out.cosh_().log_(), dim=-1)

class attention(nn.Module):
    def __init__(self, n_hidden, n_batch):
        super().__init__()
        self.n_batch = n_batch
        self.Q = nn.Linear(n_hidden, n_hidden, bias=False)
        self.K = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V = nn.Linear(n_hidden, n_hidden, bias=False)

    def forward(self, x):
        xb = x.reshape([*x.shape, self.n_batch])
        QKT = self.Q(x)
        print(QKT.shape)
        # attention = torch.nn.Softmax(dim=-1)(QKT) @ self.V(x)
        exit()
        return attention