import torch
from tqdm import trange
from ansatz.RBM import RBM
from ansatz.Jastrow import JST
from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler
from vmc.minSR import minSR
from vmc.SR import SR
from vmc.GD import GD
import numpy as np
from exact.iterator import ExactBlock
import matplotlib.pyplot as plt
from aux.colored_line import colored_line


def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))


def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(-(N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return -1 / N * np.sum(energies_p_modes)


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
    n_epoch = 100  # variational iterations
    g = 1  # Zeeman interaction amplitude
    eta = 0.001  # learning rate

    wf = JST(n_spins, dtype=dtype, device=device)
    # wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(wf.n_param)

    with Profiler(interval=0.1) as profiler:
        block = ExactBlock(
            wf, local_energy=lambda x, y: local_energy(x, y, J=-1, h=-g), verbose=0
        )

        ib = 1
        n_grid = 64
        dj = ib / n_grid
        y, x = np.mgrid[-ib:ib+dj:dj, -ib:ib+dj:dj]

        Elocs = []
        xs = x.flatten()
        ys = y.flatten()
        for n in trange(len(xs)):
            wf.assign_params(
                torch.tensor([xs[n], ys[n]], device=wf.device, dtype=wf.dtype)
            )
            Eloc = 0
            probsum = 0
            for m in range(block.configurations.shape[0]):
                a = torch.exp(wf.logprob(block.configurations[m]))
                prob = a * a.conj()
                probsum += prob
                Eloc += prob * local_energy(wf, block.configurations[m], J=-1, h=-g)
            Elocs.append(Eloc.detach().item() / probsum.detach().item())

        fig, ax = plt.subplots(1, 1)
        p = ax.pcolor(
            x,
            y,
            np.asarray(Elocs).reshape((int(np.sqrt(len(xs))), int(np.sqrt(len(xs)))))
            - min(Elocs) + 1e-1
            # - ground_state_energy_per_site(g, n_spins)
            ,
        )
        fig.colorbar(p, ax=ax)

        ib1 = 0.97
        n_grid = 2
        dj = ib1 / n_grid
        y, x = np.mgrid[-ib1:ib1+dj:dj, -ib1:ib1+dj:dj]
        xs = x.flatten()
        ys = y.flatten()
        cmaps = ['plasma', 'GnBu']
        olbl = ['GD', 'SR']
        col = ['black', 'white']
        ctr = 0
        for n in range(len(xs)):
            for (ido, optimizer) in enumerate([GD, SR]):
                line_color = [0]
                wf.assign_params(
                    torch.tensor([xs[n], ys[n]], device=wf.device, dtype=wf.dtype)
                )
                j1s = [wf.j.detach().numpy()[0]]
                j2s = [wf.j.detach().numpy()[1]]

                epochbar = trange(n_epoch)
                Eavs = torch.zeros(n_epoch, dtype=dtype)
                E_var = 1.0
                for epoch in epochbar:
                    block.compute_exact(wf)

                    Eav = block.EL @ block.probs
                    Okm = block.probs @ block.OK

                    Okbar = torch.einsum(
                        "ij,i->ij", block.OK - Okm[None, :], block.probs**0.5
                    )
                    Okm = None  # free memory
                    epsbar = (block.EL - Eav) * block.probs**0.5

                    S = Okbar.conj().T @ Okbar
                    U, S, Vd = torch.linalg.svd(S)
                    line_color.append(min(1/S).detach().numpy())


                    wf.update_params(optimizer(Okbar, epsbar, eta, thresh=1e-6))
                    j1s.append(wf.j.detach().numpy()[0])
                    j2s.append(wf.j.detach().numpy()[1])

                    E_var = torch.conj(epsbar) @ epsbar
                    # Eavs[epoch] = Eav  # for some reason, this causes a memory leak...
                    # eta /= 2 if (epoch % 50 == 0) and epoch > 0 else 1
                    epochbar.set_description(
                        f"Epoch {epoch}, Average energy {Eav.detach() / n_spins}, Energy variance {E_var.detach()}, eta {eta}"
                    )
                if ctr < 2:
                    ax.plot(j1s, j2s, color=col[ido], label=olbl[ido])
                    # lines = colored_line(j1s, j2s, line_color, ax, linewidth=2, cmap=cmaps[ido], label=olbl[ido])
                else:
                    ax.plot(j1s, j2s, color=col[ido])
                    # lines = colored_line(j1s, j2s, line_color, ax, linewidth=2, cmap=cmaps[ido])
                ax.arrow(j1s[len(j1s)//2], j2s[len(j2s)//2], j1s[len(j1s)//2+1]-j1s[len(j1s)//2], j2s[len(j2s)//2+1]-j2s[len(j2s)//2], color=col[ido], head_width=0.025, head_length=0.025)
                ax.arrow(j1s[len(j1s)//2], j2s[len(j2s)//2], j1s[len(j1s)//2+1]-j1s[len(j1s)//2], j2s[len(j2s)//2+1]-j2s[len(j2s)//2], color=col[ido], head_width=0.025, head_length=0.025)
                ctr += 1

        wf.assign_all_logprobs(block.configurations)
        a = torch.exp(wf.j[0] * wf.nn + wf.j[1] * wf.nnn)
        pt = a * a.conj()
        pt /= pt.sum()

        ic(block.configurations[pt > 0.4, :])
    ax.set_ylim(-ib, ib)
    ax.set_xlim(-ib, ib)
    plt.legend()
    plt.savefig('gd_vs_sr.png', dpi=600, bbox_inches='tight')

ic((Eav.detach() / n_spins - ground_state_energy_per_site(g, n_spins)))
