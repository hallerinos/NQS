from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call

class FFNN(nn.Module):
    def __init__(self, n_spins, n_hidden, n_layer=1, require_grad=False):
        super().__init__()
        self.n_spins = n_spins
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(n_spins, n_hidden)
        )
        for _ in range(n_layer-1):
            self.stack.append(
                nn.Sigmoid()
            )
            self.stack.append(
                nn.Linear(n_hidden, n_hidden)
            )
        self.bs = nn.Parameter(torch.rand(n_spins))
        self.set_require_grad(require_grad)
        self.n_param = sum(p.numel() for p in self.parameters())

    # disable the require_grad flag
    def set_require_grad(self, require_grad):
        for name, param in self.state_dict().items():
            param.requires_grad_(require_grad)

    def logprob(self, params, x):
        f = lambda primals: functional_call(self.forward, primals, x)
        return f(params)

    def forward(self, x):
        out = self.stack(x)
        return x @ self.bs + torch.sum((2*out.cosh_()).log_(), dim=-1)

    def probratio(self, x_nom, x_denom):
        x_diff = x_nom - x_denom
        f_nom = self.stack(x_nom)
        f_denom = self.stack(x_denom)
        log_diff = torch.log(torch.cosh(f_nom)) - torch.log(torch.cosh(f_denom))
        return torch.exp(self.bs @ x_diff + torch.sum(log_diff))

    # inplace update of parameters
    def add_(self, new):
        for (old, new) in zip(self.state_dict().items(), new.items()):
            old[1].add_(new[1])

if __name__ == "__main__":
    torch.set_default_device('cuda')
    model = FFNN(4, 8, 1)
    x = 2.0*torch.randint(0, 2, (4,)) - 1.0

    f = lambda primals: functional_call(model, primals, x)
    vjpout = vjp(f, model.state_dict())
    vjpfn_out = vjpout[1](torch.tensor(1))[0]
    grad_out = torch.func.grad(f)(model.state_dict())
    for (vp,gp) in zip(vjpfn_out.items(), grad_out.items()):
        ic(torch.allclose(vp[1], gp[1]))
