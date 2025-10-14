from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call


class ViT(nn.Module):
    def __init__(self, n_batch, n_spin, d_embed, n_patch, device='cuda', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_spin = n_spin
        self.d_embed = d_embed
        self.n_batch = n_batch
        self.n_patch = n_patch
        self.d_patch = n_spin//n_patch
        self.d_concat = self.d_patch*d_embed

        self.stack = nn.Sequential(
            make_patches(self.n_batch, self.n_patch, self.d_patch),
            nn.Linear(self.n_patch, self.d_embed, dtype=self.dtype, device=self.device),
            # scaled_dot_product_attention(self.d_embed, self.device, self.dtype),
            factored_attention(self.d_patch, self.d_embed, self.device, self.dtype),
            # factored_attention(self.d_patch, self.d_embed, self.device, self.dtype),
            # factored_attention(self.d_patch, self.d_embed, self.device, self.dtype),
            concat(self.n_batch, self.d_concat),
            nn.Linear(self.d_concat, self.d_concat, dtype=self.dtype, device=self.device),
        )

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        out = self.stack(x)
        return torch.sum(out.cosh_().log_(), dim=-1)

class make_patches(nn.Module):
    def __init__(self, n_batch, n_patch, d_patch):
        super().__init__()
        self.n_batch = n_batch
        self.d_patch = d_patch
        self.n_patch = n_patch

    def forward(self, x):
        patches = torch.reshape(x, (self.n_batch, self.d_patch, self.n_patch))
        return patches

class concat(nn.Module):
    def __init__(self, n_batch, d_concat):
        super().__init__()
        self.n_batch = n_batch
        self.d_concat = d_concat

    def forward(self, x):
        out = torch.reshape(x, (self.n_batch, self.d_concat))
        return out

class scaled_dot_product_attention(nn.Module):
    def __init__(self, d_embed, device, dtype):
        super().__init__()
        self.sqrtd = d_embed**0.5
        self.Q = nn.Linear(d_embed, d_embed, bias=False, device=device, dtype=dtype)
        self.K = nn.Linear(d_embed, d_embed, bias=False, device=device, dtype=dtype)
        self.V = nn.Linear(d_embed, d_embed, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        Qx = self.Q(x)
        Kx = self.K(x)
        Vx = self.V(x)
        QK = torch.einsum('ijk,ilk->ijl', Qx, Kx.conj()) / self.sqrtd
        QKsm = torch.nn.Softmax(dim=-1)(QK)
        attention = torch.einsum('ijk,ikl->ijl', QKsm, Vx)
        return attention

class factored_attention(nn.Module):
    def __init__(self, d_patch, d_embed, device, dtype):
        super().__init__()
        self.d_embed = d_embed
        self.d_patch = d_patch
        self.alpha = nn.Parameter((self.d_embed)**(-0.5)*torch.randn(d_patch, d_patch, device=device, dtype=dtype))
        self.V = nn.Linear(d_embed, d_embed, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        Vx = self.V(x)
        attention = torch.einsum('jk,ikl->ijl', self.alpha, Vx)
        return attention

if __name__ == "__main__":
    n_spin = 128
    n_batch = 16
    n_patch = 8
    d_embed = 32
    configs = torch.randint(0, 2, (n_batch, n_spin)).to(torch.float32)
    model = ViT(n_batch, n_spin, d_embed, n_patch)
    ic(model)
    ic(model(configs).shape)