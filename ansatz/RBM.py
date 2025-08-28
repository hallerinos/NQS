import torch
from icecream import ic

@torch.compile(fullgraph=True)
class RBM(torch.nn.Module):
    def __init__(self, n_spins, n_hidden, dtype=torch.float64, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_spins = n_spins
        self.n_hidden = n_hidden
        self.n_param = n_spins + n_hidden + n_hidden * n_spins
        self.gradients = None
        self.reset_params(device)
    
    def get_params(self):
        return torch.cat((self.b, self.c, self.W.flatten()))

    def reset_params(self, device):
        b = torch.randn(self.n_spins, dtype=self.dtype, device=device)
        c = torch.randn(self.n_hidden, dtype=self.dtype, device=device)
        W = torch.randn((self.n_hidden, self.n_spins), dtype=self.dtype, device=device)

        rn = torch.norm(b)

        self.b = (b/rn).detach().clone().requires_grad_()
        self.c = (c/rn).detach().clone().requires_grad_()
        self.W = (W/rn).detach().clone().requires_grad_()

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
        self.b.grad = torch.zeros_like(self.b)
        self.c.grad = torch.zeros_like(self.c)
        self.W.grad = torch.zeros_like(self.W)

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
    def bassign_derivatives(self, x):
        theta = self.c + torch.einsum('ij,kj->ki', self.W, x)
        Ob = x
        Oc = torch.tanh(theta)
        OW = torch.einsum('kl,km->klm', Oc, x)

        self.gradients = torch.cat((Ob, Oc, OW.flatten(start_dim=1)), dim=1)

    def assign_gradients(self):
        self.gradients = torch.cat((self.b.grad, self.c.grad, self.W.grad.flatten()))

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
        x_diff = x_nom - x_denom
        phi_nom = self.c + self.W @ x_nom
        f_nom = torch.cosh(phi_nom)
        phi_denom = self.c + self.W @ x_denom
        f_denom = torch.cosh(phi_denom)
        log_diff = torch.log(f_nom) - torch.log(f_denom)
        return torch.exp(self.b @ x_diff + torch.sum(log_diff))

    @torch.compile(fullgraph=True)
    def probratio_(self, x_nom, x_denom):
        c_tp = self.c.repeat(len(x_nom), 1).T
        x_diff = x_nom - x_denom
        phi_nom = c_tp + self.W @ x_nom.T
        f_nom = torch.cosh(phi_nom)
        phi_denom = c_tp + self.W @ x_denom.T
        f_denom = torch.cosh(phi_denom)
        log_diff = torch.log(f_nom) - torch.log(f_denom)
        val = x_diff @ self.b + torch.sum(log_diff, dim=0) 
        val = val.detach()  # without this line we have a memory leak ???
        return torch.exp(val)
