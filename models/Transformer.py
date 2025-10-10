from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call


class ViT(nn.Module):
    def __init__(self, n_batch, n_spin, d_embed, n_patch):
        super().__init__()
        self.n_spin = n_spin
        self.d_embed = d_embed

        self.stack = nn.Sequential(
            make_patches(n_patch, n_spin, n_batch),
            nn.Linear(n_patch, d_embed),
            # attention(n_spin, d_embed),
            # nn.Linear(d_embed, d_embed),
        )

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        out = self.stack(x)
        ic(out.shape)
        return torch.sum(out.cosh_().log_(), dim=-1)

class make_patches(nn.Module):
    def __init__(self, n_patch, n_spin, n_batch):
        super().__init__()
        self.n_spin = n_spin
        self.n_batch = n_batch
        self.n_patch = n_patch

    def forward(self, x):
        patches = torch.reshape(x, (self.n_batch, self.n_spin//self.n_patch, self.n_patch))
        return patches

class attention(nn.Module):
    def __init__(self, d_embed, n_batch):
        super().__init__()
        self.n_batch = n_batch
        self.Q = nn.Linear(d_embed, d_embed, bias=False)
        self.K = nn.Linear(d_embed, d_embed, bias=False)
        self.V = nn.Linear(d_embed, d_embed, bias=False)

    def forward(self, x):
        QKT = self.Q(x)
        print(QKT.shape)
        # attention = torch.nn.Softmax(dim=-1)(QKT) @ self.V(x)
        exit()
        return attention

if __name__ == "__main__":
    n_spin = 12
    n_batch = 5
    n_patch = 2
    d_embed = 4
    configs = torch.randint(0, 2, (n_batch, n_spin)).to(torch.float32)
    model = ViT(n_batch, n_spin, d_embed, n_patch)
    ic(model)
    ic(model(configs).shape)