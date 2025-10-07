from torch import nn
import torch
from icecream import ic

class RNN(nn.Module):
    def __init__(self, n_spin, hidden_size, output_size=1, dtype=torch.float64, device='cuda'):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_spin = n_spin
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.stack = nn.Sequential(
            nn.Linear(n_spin, hidden_size, dtype=self.dtype, device=self.device),
            # nn.ReLU(),
            # nn.Linear(hidden_size, output_size, dtype=self.dtype, device=self.device)
        )

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())



    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.sum(self.stack(x).cosh_(), dim=-1)
        return out.log_()