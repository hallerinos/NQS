import torch


class RBM:
    def __init__(self, n_spins, n_hidden, dtype=torch.float64, device="cpu"):
        self.device = device
        self.dtype = dtype
        self.n_spins = n_spins
        self.n_hidden = n_hidden
        self.n_param = n_spins + n_hidden + n_hidden * n_spins
        self.reset_params(device)
    
    def get_params(self):
        return torch.cat((self.b, self.c, self.W.flatten()))

    def reset_params(self, device):
        b = torch.randn(self.n_spins, dtype=self.dtype, device=device)
        c = torch.randn(self.n_hidden, dtype=self.dtype, device=device)
        W = torch.randn((self.n_hidden, self.n_spins), dtype=self.dtype, device=device)

        rn = torch.norm(b)

        self.b = b.detach().clone().requires_grad_()/rn
        self.c = c.detach().clone().requires_grad_()/rn
        self.W = W.detach().clone().requires_grad_()/rn

    @torch.compile(fullgraph=True)
    def update_params(self, all_params):
        b = all_params[: self.n_spins]
        c = all_params[self.n_spins : self.n_spins + self.n_hidden]
        W = torch.reshape(
            all_params[self.n_spins + self.n_hidden :], (self.n_hidden, self.n_spins)
        )

        with torch.no_grad():
            self.b += b
            renorm = self.b.norm()
            self.b /= renorm
            self.c += c
            self.c /= renorm
            self.W += W
            self.W /= renorm
            self.reset_gattr()

    @torch.compile(fullgraph=True)
    def reset_gattr(self):
        self.b.grad = None
        self.c.grad = None
        self.W.grad = None

    @torch.compile(fullgraph=True)
    def assign_derivatives(self, x):
        theta = self.c + self.W @ x
        Ob = x
        Oc = torch.tanh(theta)
        OW = Oc[:, None] @ x[None, :]

        self.Ob = Ob
        self.Oc = Oc
        self.OW = OW

    @torch.compile(fullgraph=True)
    def prob(self, x):
        return torch.exp(self.b @ x) * torch.prod(2 * torch.cosh(self.c + self.W @ x))

    @torch.compile(fullgraph=True)
    def prob_(self, x):
        return torch.exp(self.b.conj() @ x) * torch.prod(
            2 * torch.cosh(self.c.conj() + self.W.conj() @ x)
        )

    @torch.compile(fullgraph=True)
    def logprob(self, x):
        return self.b @ x + torch.sum(torch.log(2 * torch.cosh(self.c + self.W @ x)))

    @torch.compile(fullgraph=True)
    def logprob_(self, x):
        return self.b.conj() @ x + torch.sum(
            torch.log(2 * torch.cosh(self.c.conj() + self.W.conj() @ x))
        )

    @torch.compile(fullgraph=True)
    def probratio(self, x_nom, x_denom):
        f_nom = torch.cosh(self.c + self.W @ x_nom)
        f_denom = torch.cosh(self.c + self.W @ x_denom)
        return torch.exp(self.b @ (x_nom - x_denom) + torch.sum(torch.log(f_nom / f_denom)))
