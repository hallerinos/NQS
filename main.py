from models.FFNN import FFNN
from models.Transformer import SDPA
import torch
from icecream import ic
from local_energies.TFIM import TFIM
from sampler.MCMC import MCMC
import tqdm
import numpy as np
from linalg.metric_tensor import S
from collections import OrderedDict
from linalg.conjugate_gradient import cg
from copy import copy, deepcopy

if __name__ == "__main__":
    n_epoch = 2**12
    n_spin = 2**4
    Ns = 2**16
    eta = 1e-2

    model = FFNN(n_spin, n_spin)
    model.requires_grad_(False)  # less memory and better performance

    ic(n_spin, Ns, model.n_param)

    sampler = MCMC(model, Ns, local_energy=lambda model, x: TFIM(model, x, J=-1, h=-1))

    tbar = tqdm.trange(n_epoch)
    dThp = copy(model.state_dict())
    for epoch in tbar:
        # sample new set of configurations
        sampler.sample(n_res=n_spin)

        # energy averages
        Eav = sampler.EL.mean()
        epsbar = (sampler.EL - Eav) / Ns**0.5
        Evar = (torch.conj(epsbar) @ epsbar).mean()

        # helper functions for gradients
        f = lambda primals: torch.func.functional_call(sampler.model, primals, sampler.samples)
        _, vjp = torch.func.vjp(f, sampler.model.state_dict())
        vjpres = vjp(sampler.EL)[0]

        fav = lambda primals: torch.func.functional_call(sampler.model, primals, sampler.samples).mean()
        _, vjpav = torch.func.vjp(fav, sampler.model.state_dict())
        vjpavres = vjpav(Eav)[0]

        # initialize metric tensor
        metric_tensor = S(f, fav, sampler.model, Ns)

        # compute Euclidean gradient
        dThd = OrderedDict()
        for key in vjpres.keys():
            dThd[key] = vjpres[key]/Ns - vjpavres[key]

        # solve S x = dThd
        x = cg(metric_tensor, dThd, dThp, max_iter=4)
        # ic(metric_tensor.compute_residual(x[0], dThd))  # residual = norm(S x[0] - dThd)

        # update parameters
        dThd = x[0]
        for (k, v) in dThd.items():
            sampler.model.state_dict()[k].add_(-eta*dThd[k])
        dThp = deepcopy(dThd)

        # update progress bar
        Edens = Eav / n_spin
        tbar.set_description(
                f"E/N: {np.round(Edens.cpu(), decimals=4)}, \u03C3\u00B2/N: {np.round(Evar.cpu()/n_spin, decimals=4)}"
            )