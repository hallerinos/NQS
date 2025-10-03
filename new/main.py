from CNN import FFNN
import torch
from icecream import ic
from energies import TFIM
from sampler import sampler

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    n_spin, Ns = 2**9, 2**14

    model = FFNN(n_spin, n_spin)

    ic(n_spin, Ns, model.n_param)
    x = (2*torch.randint(0, 2, (Ns, n_spin)) - 1).to(torch.get_default_device(), torch.get_default_dtype())

    # TFIM(model, x)
    sampler = sampler(model, Ns, local_energy=TFIM)
    with torch.no_grad():
        sampler.warmup(n_res=n_spin//4)