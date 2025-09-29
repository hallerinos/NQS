import torch
from icecream import ic

# @torch.compile(fullgraph=False)
class MF(torch.nn.Module):
    def __init__(self, n_spins, dtype=torch.float64, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_spins = n_spins
        self.n_param = 2*n_spins
        self.gradients = None
        self.reset_params(device)
    
    def get_params(self):
        return torch.cat((self.phi, self.theta))

    def reset_params(self, device):
        self.theta = torch.pi*torch.rand(self.n_spins, dtype=self.dtype, device=device)
        self.phi = 2*torch.pi*(torch.rand(self.n_spins, dtype=self.dtype, device=device) - 0.5)

    def update_params(self, all_params):
        with torch.no_grad():
            theta = all_params[:self.n_spins]
            phi = all_params[self.n_spins:]

            self.theta += theta
            self.phi += phi

    # @torch.compile(fullgraph=False)
    def logprob(self, x):
        return torch.log(torch.exp(1j/2*x*self.phi) * torch.sin(self.theta/2 + (x+1) * torch.pi/4)).sum()

    # @torch.compile(fullgraph=False)
    def logprob(self, x, angles):
        return torch.log(torch.exp(1j/2*x*angles[:self.n_spins]) * torch.sin(angles[self.n_spins:]/2 + (x+1) * torch.pi/4)).sum()

    # @torch.compile(fullgraph=False)
    def probratio(self, x_nom, x_denom):
        res = torch.where(x_nom == x_denom, 1.0, torch.exp(1j/2*(x_nom-x_denom)*self.phi)*torch.sin(self.theta/2 + (x_nom+1) * torch.pi/4 + 1e-3) / torch.sin(self.theta/2 + (x_denom+1) * torch.pi/4 + 1e-3)).prod()
        return res