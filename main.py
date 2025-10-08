from models.RNN import RNN
from models.FFNN import FFNN
from models.Transformer import SDPA
import torch
from icecream import ic
from local_energies.TFIM import TFIM, TFIM_rot
from local_energies.references import ground_state_energy_per_site
from sampler.MCMC import MCMC
import tqdm
import numpy as np
from linalg.metric_tensor import S
from collections import OrderedDict
from linalg.conjugate_gradient import cg
from copy import copy, deepcopy

if __name__ == "__main__":
    n_epoch = 2**14
    n_spin = 2**6
    Ns = 2**14
    eta = 0.01
    g = 1
    # torch.manual_seed(0)

    # model = RNN(n_spin, n_spin)
    model = FFNN(n_spin, n_spin, device='cuda', dtype=torch.double)
    model.requires_grad_(False)  # less memory and better performance


    E_exact = ground_state_energy_per_site(g, n_spin)
    ic(E_exact)

    ic(n_spin, Ns, model.n_param)

    sampler = MCMC(model, Ns, local_energy=lambda model, x: TFIM(model, x, J=-1, h=-g))

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
        metric_tensor = S(f, fav, sampler.model, Ns, diag_reg=1e-2)

        # compute Euclidean gradient dE/dTheta
        dEdTh = OrderedDict()
        for key in vjpres.keys():
            dEdTh[key] = vjpres[key]/Ns - vjpavres[key]

        # solve the SR equation S dTh = dEdTh
        dTh, _ = cg(metric_tensor, dEdTh, dThp, max_iter=4)

        # update parameters
        for (k, v) in dTh.items():
            sampler.model.state_dict()[k].add_(-eta*dTh[k])
        dThp = copy(dTh)

        # update progress bar
        Edens = Eav
        tbar.set_description(
                f"E/N: {np.round(Edens.cpu(), decimals=4)}, \u03C3\u00B2/N: {np.round(Evar.cpu(), decimals=4)}"
            )