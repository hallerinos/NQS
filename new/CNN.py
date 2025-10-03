from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call


class FFNN(nn.Module):
    def __init__(self, n_spins, n_hidden, n_layer=1):
        super().__init__()
        self.n_spins = n_spins
        self.n_hidden = n_hidden
        self.n_layer = n_layer

        # initialize the layers
        self.input_bias = nn.Parameter(torch.rand(n_spins))
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

        # compute number of parameters
        self.n_param = sum(p.numel() for p in self.parameters())

    # returns logprob
    def forward(self, x):
        out = self.stack(x)
        return x @ self.input_bias + torch.sum((2*out.cosh_()).log_(), dim=-1)

    # returns probability ratio p_nom(x_nom) / p_denom(x_denom) given two configurations x_nom, x_denom
    def probratio(self, x_nom, x_denom):
        x_diff = x_nom - x_denom
        f_nom = self.stack(x_nom)
        f_denom = self.stack(x_denom)
        log_diff = f_nom.cosh_().log_() - f_denom.cosh_().log_()
        return torch.exp(x_diff @ self.input_bias + torch.sum(log_diff, dim=-1))

    # define logprob given input x evaluated for parameters params
    def logprob(self, params, x):
        return functional_call(self, params, x)

    # return vectorized version of all parameters
    def get_parameters_all(self):
        return torch.nn.utils.parameters_to_vector(self.parameters())

    # set parameters from vector
    def set_parameters_all(self, vector):
        torch.nn.utils.vector_to_parameters(vector, self.parameters())



if __name__ == "__main__":
    torch.set_default_device('cuda')
    model = FFNN(4, 8, 1)
    ic(model.n_param)
    params = model.get_parameters_all()
    model.set_parameters_all(0*params)
    ic(model.state_dict())
    x = 2.0*torch.randint(0, 2, (4,)) - 1.0

    def f(primals): return functional_call(model, primals, x)
    vjpout = vjp(f, model.state_dict())
    vjpfn_out = vjpout[1](torch.tensor(1))[0]
    grad_out = torch.func.grad(f)(model.state_dict())
    for (vp, gp) in zip(vjpfn_out.items(), grad_out.items()):
        ic(torch.allclose(vp[1], gp[1]))
