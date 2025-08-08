import torch


class MPS:
    def __init__(self, n_spins, bond_dim, dtype=torch.float64, device="cpu"):
        self.device = device
        self.dtype = dtype
        self.n_spins = n_spins
        self.bond_dim = bond_dim
        self.n_param = n_spins*(2*bond_dim**2)
        self.reset_params(device)

    def reset_params(self, device):
        A = torch.randn((self.n_spins, self.bond_dim, 2, self.bond_dim), dtype=self.dtype, device=device)

        self.A = A.detach().clone().requires_grad_()

    # @torch.compile(fullgraph=True)
    def update_params(self, all_params):
        with torch.no_grad():
            self.A[:] += all_params[:]
            renorm = self.A.norm()
            self.A /= renorm
            self.A.grad = None

    # @torch.compile(fullgraph=True)
    def reset_gattr(self):
        self.A.grad = None

    # @torch.compile(fullgraph=True)
    def prob(self, x):
        pamp = torch.eye(self.bond_dim)
        for ni in range(self.n_spins):
            pamp *= self.A[ni, :, x[ni], :]
        return pamp

    # @torch.compile(fullgraph=True)
    def prob_(self, x):
        return torch.exp(self.b.conj() @ x) * torch.prod(
            2 * torch.cosh(self.c.conj() + self.W.conj() @ x)
        )

    # @torch.compile(fullgraph=True)
    def logprob(self, x):
        return self.b @ x + torch.sum(torch.log(2 * torch.cosh(self.c + self.W @ x)))

    # @torch.compile(fullgraph=True)
    def logprob_(self, x):
        return self.b.conj() @ x + torch.sum(
            torch.log(2 * torch.cosh(self.c.conj() + self.W.conj() @ x))
        )

    # @torch.compile(fullgraph=True)
    def probratio(self, x_nom, x_denom):
        f_nom = torch.cosh(self.c + self.W @ x_nom)
        f_denom = torch.cosh(self.c + self.W @ x_denom)
        return torch.exp(self.b @ (x_nom - x_denom) + torch.sum(torch.log(f_nom / f_denom)))
