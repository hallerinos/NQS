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

def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))

def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)

@torch.compile(fullgraph=True)
def local_energy(wf: RBM, spin_vector: torch.Tensor, J=-1, h=-1):
    interactions = spin_vector[1:] @ spin_vector[:-1]
    interactions += spin_vector[0] * spin_vector[-1]  # pbc


    # Create copies of original configuration and a batch of flipped configurations
    spin_vector_r = spin_vector.repeat(len(spin_vector), 1)
    flipped_spins = spin_vector.repeat(len(spin_vector), 1)
    # Create indices for diagonal elements
    indices = torch.arange(len(spin_vector), device=spin_vector.device)
    # Flip spins in batch
    flipped_spins[indices, indices] *= -1
    # Calculate all probability ratios at once
    zeeman_term = wf.probratio_(flipped_spins, spin_vector_r).sum()

    return J * interactions + h * zeeman_term


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.double

    print(torch.__version__)

    n_spins = 100  # spin sites
    alpha = 1
    n_hidden = alpha * n_spins  # neurons in hidden layer
    n_block = 2**10  # samples / expval
    n_epoch = 100  # variational iterations
    g = 1.0  # Zeeman interaction amplitude
    eta = 0.01  # learning rate

    # wf = JST(n_spins, dtype=dtype, device=device)
    wf = torch.compile(RBM(n_spins, n_hidden, dtype=dtype, device=device))
    ic(wf.n_param)

    epochbar = trange(n_epoch)
    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    with Profiler(interval=0.1) as profiler:
        block = MCBlock(wf, n_block, local_energy=lambda x, y: local_energy(x, y, J=-1, h=-g))
        for epoch in epochbar:
            block.sample_block(wf, n_block)

            Eav = torch.mean(block.EL, dim=0)
            Okm = torch.mean(block.OK, dim=0)

            Okbar = (block.OK - Okm[None, :]) / n_block**0.5
            Okm = None  # free memory
            epsbar = (block.EL - Eav) / n_block**0.5

            wf.update_params(minSR(Okbar, epsbar, torch.tensor(eta, dtype=wf.dtype, device=wf.device)))

            E_var = torch.conj(epsbar) @ epsbar
            # Eavs[epoch] = Eav  # for some reason, this causes a memory leak...
            # eta /= 2 if (epoch % 50 == 0) and epoch > 0 else 1
            epochbar.set_description(
                f"Epoch {epoch}, E {Eav.detach()/n_spins}, V {E_var.detach()}"
            )
    profiler.write_html('our_profiler.html')

ic((Eav.detach()/n_spins - ground_state_energy_per_site(g, n_spins)))
