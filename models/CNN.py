from torch import nn
import torch
from icecream import ic

class CNN(nn.Module):
    def __init__(self, n_spin, n_hidden, n_layer=1, dtype=torch.float64, device='cuda'):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_spin = n_spin
        self.n_hidden = n_hidden
        self.n_layer = n_layer

        # initialize the layers
        self.stack = nn.Sequential(
            nn.Linear(n_spin, n_hidden, dtype=self.dtype, device=self.device),
            torch.nn.ELU()
        )
        for _ in range(n_layer-1):
            self.stack.append(
                nn.Tanh()
            )
            self.stack.append(
                nn.Linear(n_hidden, n_hidden, dtype=self.dtype, device=self.device)
            )

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())

    # evaluates log(wavefunction)
    def forward(self, x):
        out = 0.0
        # sum all possible translations of the spin configuration
        for i in range(x.size(-1)):
            out += self.stack(torch.roll(x, i, dims=-1))

        return torch.sum(out, dim=-1)

    # returns probability ratio log(wavefunction(x_nom)) / log(wavefunction(x_denom))
    def probratio(self, x_nom, x_denom):
        f_nom = self.stack(x_nom)
        f_denom = self.stack(x_denom)
        log_diff = f_nom.cosh_().log_() - f_denom.cosh_().log_()
        return torch.exp(torch.sum(log_diff, dim=-1))