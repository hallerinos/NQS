import torch
from tqdm import trange
from ansatz.RBM import RBM
from ansatz.Jastrow import JST
from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler
from vmc.minSR import minSR
from vmc.SR import SR, SR_, SR__
import numpy as np
from energies.TFIM import local_energy
# torch.manual_seed(0)
from torch import Tensor
from linalg.cg import cg, bicgstab
from linalg.kernels import S
import linear_operator

def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))

def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float32

    print(torch.__version__)

    n_spins = 2**4  # spin sites
    alpha = 1
    n_hidden = int(alpha * n_spins)  # neurons in hidden layer
    n_block = 2**10  # samples / expval
    n_epoch = 2**10  # variational iterations
    g = 10.0  # Zeeman interaction amplitude
    eta = 0.01  # learning rate

    E_exact = ground_state_energy_per_site(g, n_spins)
    ic(E_exact)

    # wf = JST(n_spins, dtype=dtype, device=device)
    wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(n_epoch, n_block, n_spins, wf.n_param)

    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    block = MCBlock(wf, n_block, local_energy=lambda x, y: local_energy(x, y, J=-1, h=-g))
    epochbar = trange(n_epoch)
    dTh = torch.zeros(wf.n_param, device=wf.device, dtype=wf.dtype)
    with Profiler(interval=0.1) as profiler:
        for epoch in epochbar:
            block.bsample_block_no_grad(wf, n_res=16)

            Eav = torch.mean(block.EL, dim=0)
            epsbar = (block.EL - Eav) / n_block**0.5

            vlogpsi = torch.vmap(wf.logprob, in_dims=(0, None))
            f = lambda *primals: vlogpsi(block.samples, *primals)
            _, vjp = torch.func.vjp(f, wf.get_params())

            fav = lambda *primals: vlogpsi(block.samples, *primals).mean()
            _, vjpav = torch.func.vjp(fav, wf.get_params())

            qmetr = S(f, fav, wf, n_block, diag_reg=0)

            dThn = vjp(block.EL)[0] / n_block - vjpav(Eav)[0]

            x = cg(qmetr, dThn, dTh, max_iter=8)
            ic((qmetr @ x[0] - dThn).norm())
            x = bicgstab(qmetr, dThn, dTh, max_iter=4)
            ic((qmetr @ x[0] - dThn).norm())
            dThn = x[0]  # next update step

            wf.update_params(-eta * dThn)
            dTh = dThn  # save for initial guess

            E_var = torch.conj(epsbar) @ epsbar
            edens = Eav.detach().cpu().item()/n_spins

            Eavs[epoch] = edens
            epochbar.set_description(
                f"E/N: {np.round(edens, decimals=4)}, \u03C3: {np.round(E_var.detach().cpu(), decimals=4)}"
            )
            if edens != edens:
                print("NaN encountered!")
                break
    profiler.write_html('our_profiler.html')
    print(Eavs)

# ic((Eav.detach()/n_spins - E_exact))
