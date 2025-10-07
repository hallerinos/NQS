import torch
from pyinstrument import Profiler
from tqdm import trange
from collections.abc import Callable
from icecream import ic

def local_energy(x: torch.Tensor, *params: torch.Tensor, J=-1, h=-1):
    interactions = x[1:] @ x[:-1]
    interactions += x[0] * x[-1]  # pbc

    zeeman_term = torch.zeros_like(interactions)
    # sequential adding energy of spin flips (less memory, more time)
    for i in range(x.shape[0]):
        x_f = x.clone()
        x_f[i] *= -1
        zeeman_term += probratio(x_f, x, *params)

    return J * interactions + h * zeeman_term

def amap(b, c, W, x):
    h = torch.cosh(c + W @ x)
    return h

def logpsi(x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
    return params[0] @ x + torch.log(amap(*params, x)).sum()

def probratio(x_nom: torch.Tensor, x_denom: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
    logratio = params[0] @ (x_nom - x_denom)
    logratio = logratio + torch.log(amap(*params, x_nom)).sum()
    logratio = logratio - torch.log(amap(*params, x_denom)).sum()
    return torch.exp(logratio)

def bicg(x0: torch.Tensor, f: Callable[[torch.Tensor], torch.NumberType]) -> torch.Tensor:
    fx0, vjp = torch.func.vjp(f, x0)
    return fx0

if __name__ == "__main__":

    nbatch, nspins, nhidden = 2**14, 2**4, 2**4
    nparam = nspins+nhidden+nhidden*nspins
    ic(nbatch, nspins, nparam)

    x0 = (2*torch.randint(0, 2, (nbatch, nspins, )) - 1.0).to('cuda')
    b = torch.rand((nspins, )).to('cuda')
    c = torch.rand((nhidden, )).to('cuda')
    W = torch.rand((nhidden, nspins, )).to('cuda')

    vEloc = torch.compile(torch.vmap(local_energy, in_dims=(0, *(3*[None]))))
    vlogpsi = torch.compile(torch.vmap(logpsi, in_dims=(0, *(3*[None]))))
    for _ in trange(100):
        f = lambda *primals: vlogpsi(x0, *primals)
        val, vjp = torch.func.vjp(f, b, c, W)
        vjp(vEloc(x0, b, c, W))