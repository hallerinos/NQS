from torch import nn
import torch
from icecream import ic

class MPS(nn.Module):
    def __init__(self, n_spin, d_bond, d_local=2, dtype=torch.float64, device='cuda'):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_spin = n_spin
        self.d_bond = d_bond
        self.d_local = d_local

        tens = torch.randn((n_spin, d_bond, d_local, d_bond), dtype=dtype, device=device)
        self.stack = torch.nn.Parameter(tens/tens.norm())

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())



    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        xind = ((x+1)/2).to(torch.int)
        out = self.stack[0, :, xind[:,0], :]
        for i in range(1, x.size(-1)):
            out = torch.einsum('ijk,kjm->ijm', out, self.stack[i, :, xind[:,i], :])
        out = torch.einsum('iji->j',out)
        return out