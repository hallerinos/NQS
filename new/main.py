from CNN import FFNN
import torch
from icecream import ic
from energies import TFIM
from sampler import sampler
import tqdm
import numpy as np
from kernels import S
from collections import OrderedDict
from cg import cg
from copy import copy, deepcopy

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    torch.set_default_device('cuda')
    # torch.manual_seed(0)

    n_spin = 2**8
    Ns = 128*n_spin
    eta = 1/n_spin

    model = FFNN(n_spin, n_spin)
    model.requires_grad_(False)

    ic(n_spin, Ns, model.n_param)

    sampler = sampler(model, Ns, local_energy=lambda model, x: TFIM(model, x, J=-1, h=-1))
    # sampler.warmup(n_res=16, max_iter=2**10)
    tbar = tqdm.trange(2**10)
    dThp = copy(model.state_dict())
    for epoch in tbar:
        sampler.sample(n_res=n_spin)

        Eav = sampler.EL.mean()
        epsbar = (sampler.EL - Eav) / Ns**0.5
        Evar = (torch.conj(epsbar) @ epsbar).mean()

        f = lambda primals: torch.func.functional_call(sampler.model, primals, sampler.samples)
        _, vjp = torch.func.vjp(f, sampler.model.state_dict())
        vjpres = vjp(sampler.EL)[0]

        fav = lambda primals: torch.func.functional_call(sampler.model, primals, sampler.samples).mean()
        _, vjpav = torch.func.vjp(fav, sampler.model.state_dict())
        vjpavres = vjpav(Eav)[0]

        qmetr = S(f, fav, sampler.model, Ns, diag_reg=1e-3)

        dThd = OrderedDict()
        for key in vjpres.keys():
            dThd[key] = vjpres[key]/Ns - vjpavres[key]

        x = cg(qmetr, dThd, dThp, max_iter=4)
        # residual = 0
        # for key in x[0].keys():
        #     nk = (qmetr @ x[0])[key] - dThd[key]
        #     residual += nk.flatten().conj() @ nk.flatten()
        # residual = residual**(0.5)
        dThd = x[0]

        for (k, v) in dThd.items():
            sampler.model.state_dict()[k].add_(-eta*dThd[k])
        dThp = deepcopy(dThd)

        Edens = Eav / n_spin
        tbar.set_description(
                f"E/N: {np.round(Edens.cpu(), decimals=4)}, \u03C3: {np.round(Evar.cpu(), decimals=4)}"
            )