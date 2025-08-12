import torch
from tqdm import trange
from ansatz.RBM import RBM
from ansatz.Jastrow import JST
from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler
from vmc.minSR import minSR
from vmc.SR import SR
import numpy as np
from exact.iterator import ExactBlock
import matplotlib.pyplot as plt

def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))

def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)

# @torch.compile(fullgraph=True)
def local_energy(wf: RBM, spin_vector: torch.Tensor, J=-1, h=-1):
    interactions = spin_vector[1:] @ spin_vector[:-1]
    interactions += spin_vector[0] * spin_vector[-1]  # pbc


    # Create batch of flipped configurations
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
    device = "cpu"
    dtype = torch.double
    # dtype = torch.complex128

    print(torch.__version__)

    n_spins = 6  # spin sites
    n_hidden = 10  # neurons in hidden layer
    n_block = 1000  # samples / expval
    n_epoch = 1000  # variational iterations
    g = 1.0  # Zeeman interaction amplitude
    eta = 0.01  # learning rate

    wf = JST(n_spins, dtype=dtype, device=device)
    wf_2 = JST(n_spins, dtype=dtype, device=device)
    # wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(wf.n_param)

    epochbar = trange(n_epoch)
    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    with Profiler(interval=0.1) as profiler:
        block = ExactBlock(wf, local_energy=lambda x, y: local_energy(x, y, J=-1, h=-g), verbose=0)

        ib = 1
        dj = ib/32
        y, x = np.mgrid[-ib:ib:dj, -ib:ib:dj]

        Elocs = []
        for (j1, j2) in zip(x.flatten(), y.flatten()):
            wf_2.assign_params(torch.tensor([j1, j2], device=wf.device, dtype=wf.dtype))
            Eloc = 0
            probsum = 0
            for conf in block.configurations:
                a = torch.exp(wf_2.logprob(conf))
                prob = a * a.conj()
                probsum += prob
                Eloc += prob * local_energy(wf_2, conf, J=-1, h=-g)
            Elocs.append(Eloc.item() / probsum.item())
        ic(Elocs)

        fig, ax = plt.subplots(1, 1)
        p = ax.pcolor(x, y, np.log(abs(np.asarray(Elocs).reshape((64,64))/n_spins - ground_state_energy_per_site(g, n_spins))))
        fig.colorbar(p, ax=ax)

        for epoch in epochbar:
            block.compute_exact(wf, n_block)

            Eav = block.EL @ block.probs
            Okm = block.probs @ block.OK

            Okbar =  torch.einsum('ij,i->ij', block.OK - Okm[None, :], block.probs**0.5)
            Okm = None  # free memory
            epsbar = (block.EL - Eav) * block.probs**0.5

            wf.update_params(minSR(Okbar, epsbar, eta, thresh=1e-6))

            E_var = torch.conj(epsbar) @ epsbar
            # Eavs[epoch] = Eav  # for some reason, this causes a memory leak...
            # eta /= 2 if (epoch % 50 == 0) and epoch > 0 else 1
            epochbar.set_description(
                f"Epoch {epoch}, Average energy {Eav.detach()/n_spins}, Energy variance {E_var.detach()}, eta {eta}"
            )

    profiler.print()

    ic(wf.j)
    wf.assign_all_logprobs(block.configurations)
    a = torch.exp(wf.j[0] * wf.nn + wf.j[1] * wf.nnn)
    pt = a * a.conj()
    pt /= pt.sum()
    ic(pt)
    ic(block.configurations[pt > 0.4, :])
    # plt.plot(np.sort(pt.detach().numpy()))
    # plt.show()
    js = wf.j.detach().numpy()
    ax.scatter(js[0], js[1])
    plt.show()

ic((Eav.detach()/n_spins - ground_state_energy_per_site(g, n_spins)))