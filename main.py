from models.RNN import RNN
from models.FFNN import FFNN
from models.CNN import CNN
import torch
from icecream import ic
from local_energies.Heisenberg import Heisenberg
from local_energies.TFIM import TFIM, TFIM_y, TFIM_rot
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
    n_spin = 2**4
    Ns = 2**14
    eta = 1e-2
    diag_reg = 1e-2
    adaptive = 1
    g = 1
    # torch.manual_seed(0)

    # model = RNN(n_spin, n_spin)
    model = FFNN(n_spin, n_spin, device='cuda', dtype=torch.complex64)
    model.requires_grad_(False)  # less memory and better performance

    ic(n_spin, Ns, model.n_param)

    # sampler = MCMC(model, Ns, local_energy=lambda model, x: Heisenberg(model, x, J=[1.0,1.0,1.0], B=[-0.0,-0.0,-0.0]))
    sampler = MCMC(model, Ns, local_energy=lambda model, x: TFIM(model, x, J=0, h=-g))

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
        diag_reg = max(adaptive * diag_reg, 1e-16)
        metric_tensor = S(f, fav, sampler.model, Ns, diag_reg=diag_reg)

        # compute Euclidean gradient dE/dTheta
        dEdTh = OrderedDict()
        for key in vjpres.keys():
            dEdTh[key] = vjpres[key]/Ns - vjpavres[key]
            # dEdTh[key].copy_(0.5*dEdTh[key].add_(dEdTh[key].conj()))
        dTh = dEdTh

        # solve the SR equation S dTh = dEdTh
        dTh, _ = cg(metric_tensor, dEdTh, dThp, tol=1e-4, max_iter=8)

        # update parameters
        for (k, v) in dTh.items():
            if dTh[k].norm() < 1e2:
                sampler.model.state_dict()[k].add_(-eta*dTh[k])
            elif dEdTh[k].norm() < 1e2:
                sampler.model.state_dict()[k].add_(-eta*dEdTh[k])
        dThp = copy(dTh)

        lnwf0 = sampler.model(sampler.samples)
        
        sx_exp = torch.zeros(sampler.samples.size(-1))
        sy_exp = torch.zeros(sampler.samples.size(-1))
        sz_exp = torch.zeros(sampler.samples.size(-1))
        for i in range(sampler.samples.size(-1)):
            spin_vector_f = sampler.samples.clone()
            spin_vector_f[:, i] *= -1
            lnwf1 = sampler.model(spin_vector_f)
            val = torch.exp(lnwf1 - lnwf0)
            sx_exp[i] = val.mean()
            sy_exp[i] = (-1j*val*(sampler.samples[:,i])).mean()
            sz_exp[i] = sampler.samples[:,i].mean()
        # ic(sx_exp, sy_exp, sz_exp)

        # update progress bar
        tbar.set_description(
                f"E/N: {Eav.cpu()/n_spin:.6e}, \u03C3\u00B2/N: {Evar.cpu()/n_spin:.2e}, diag_reg: {diag_reg:.2e}"
            )