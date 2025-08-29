import torch
from tqdm import trange
from ansatz.RBM import RBM
from ansatz.Jastrow import JST
from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler
from vmc.minSR import minSR
from vmc.SR import SR, SR_
import numpy as np
from energies.TFIM import local_energy
# torch.manual_seed(0)


def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))

def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.double

    print(torch.__version__)

    n_spins = 2**6  # spin sites
    alpha = 1
    n_hidden = alpha * n_spins  # neurons in hidden layer
    n_block = 2**14  # samples / expval
    n_epoch = 2**14  # variational iterations
    g = 1.0  # Zeeman interaction amplitude
    eta = torch.tensor(0.01, device=device, dtype=dtype)  # learning rate

    E_exact = ground_state_energy_per_site(g, n_spins)
    ic(E_exact)

    # wf = JST(n_spins, dtype=dtype, device=device)
    wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(n_epoch, n_block, wf.n_param)

    epochbar = trange(n_epoch)
    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    with Profiler(interval=0.1) as profiler:
        block = MCBlock(wf, n_block, local_energy=lambda x, y: local_energy(x, y, J=-1, h=-g))
        block.bsample_block(wf, n_block, n_iter=n_spins)
        for epoch in epochbar:
            block.bsample_block(wf, n_block, n_iter=4)

            # continue

            Eav = torch.mean(block.EL, dim=0)
            Okm = torch.mean(block.OK, dim=0)

            Okbar = (block.OK - Okm) / n_block**0.5
            # Okm = None  # free memory
            epsbar = (block.EL - Eav) / n_block**0.5

            # dTh = SR(Okbar, epsbar, diag_reg=1e-4)
            dTh = Okbar.conj().T @ epsbar
            wf.update_params(-eta * dTh)

            E_var = torch.conj(epsbar) @ epsbar
            # Eavs[epoch] = Eav.detach().item()  # for some reason, this causes a memory leak...
            epochbar.set_description(
                f"Epoch {epoch}, E {Eav.detach()/n_spins}, V {E_var.detach()}"
            )
    profiler.write_html('our_profiler.html')

# ic((Eav.detach()/n_spins - E_exact))
