from CNN import FFNN
import torch
from icecream import ic
from energies import TFIM
from sampler import sampler
import tqdm
import numpy as np

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    n_spin, Ns, eta = 2**6, 2**14, 0.01

    model = FFNN(n_spin, n_spin)
    model.requires_grad_(False)

    ic(n_spin, Ns, model.n_param)

    sampler = sampler(model, Ns, local_energy=TFIM)
    sampler.warmup(n_res=n_spin//4, max_iter=2**10)
    tbar = tqdm.trange(2**8)
    for epoch in tbar:
        sampler.sample(n_res=n_spin//4)

        Eav = sampler.EL.mean()
        epsbar = (sampler.EL - Eav) / Ns**0.5
        Evar = torch.conj(epsbar) @ epsbar

        f = lambda primals: torch.func.functional_call(model, primals, sampler.samples)
        _, vjp = torch.func.vjp(f, model.state_dict())
        vjpres = vjp(sampler.EL)[0]

        fav = lambda primals: torch.func.functional_call(model, primals, sampler.samples).mean()
        _, vjpav = torch.func.vjp(fav, model.state_dict())
        vjpavres = vjpav(Eav)[0]
        for (k, v) in vjpres.items():
            dTh = v/Ns - vjpavres[k]
            sampler.model.state_dict()[k] += - eta  * dTh
        Edens = Eav / n_spin
        tbar.set_description(
                f"E/N: {np.round(Edens.cpu(), decimals=4)}, \u03C3: {np.round(Evar.mean().cpu(), decimals=4)}"
            )