import torch

# Jastrow Ansatz for the 1D Ising model
class JST:
    def __init__(self, n_spins, dtype=torch.float64, device='cpu'):
        self.device = device
        self.dtype = dtype
        self.n_param = 2
        self.n_spins = n_spins
        self.gradients = None
        self.reset_params()

    def reset_params(self):
        j = torch.randn(2, dtype=self.dtype, device=self.device)
        self.j = (j / j.norm()).detach().clone().requires_grad_()
    
    def reset_gattr(self):
        self.j.grad = None

    def assign_params(self, all_params):
        j = all_params[:]
        self.j = j / j.norm()

    def update_params(self, all_params):
        j = all_params

        with torch.no_grad():
            self.j += j
            self.j.grad = None

    def assign_derivatives(self, x):
        self.Oj = torch.tensor([x.roll(1) @ x, x.roll(2) @ x], dtype=self.dtype, device=self.device)
    
    def assign_gradients(self):
        self.gradients = self.j.grad

    def logprob(self, x):
        nn = self.j[0] * x.roll(1) @ x
        nnn = self.j[1] * x.roll(2) @ x

        return nn + nnn

    def probratio(self, x_nom, x_denom):
        f_nom = self.j[0] * x_nom.roll(1) @ x_nom + self.j[1] * x_nom.roll(2) @ x_nom
        f_denom = self.j[0] * x_denom.roll(1) @ x_denom + self.j[1] * x_denom.roll(2) @ x_denom
        return torch.exp(f_nom - f_denom)
    
    def probratio_(self, x_nom, x_denom):
        f_nom = self.j[0] * x_nom.roll(1) @ x_nom + self.j[1] * x_nom.roll(2) @ x_nom
        f_denom = self.j[0] * x_denom.roll(1) @ x_denom + self.j[1] * x_denom.roll(2) @ x_denom
        return torch.exp(f_nom - f_denom)