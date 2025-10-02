from torch import nn
import torch
from icecream import ic
from torch.func import vjp, functional_call

class NeuralNetwork(nn.Module):
    def __init__(self, n_spin, n_hidden, n_layer=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(n_spin, n_hidden)
        )
        for _ in range(n_layer-1):
            self.stack.append(nn.Linear(n_hidden, n_hidden))

    def forward(self, x):
        x = self.flatten(x)
        out = self.stack(x)
        return out


if __name__ == "__main__":
    torch.set_default_device('cuda')
    model = NeuralNetwork(4, 8, 2).to('cuda')

    x = 2.0*torch.randint(0, 2, (10, 4)) - 1.0
    f = lambda params: functional_call(model, params, x)
    fp, fvjp = vjp(f, dict(model.named_parameters()))

    c = 1.0*torch.randint(0, 2, (10,8))
    ic(fvjp(c))