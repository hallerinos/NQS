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
from linalg.cg import cg

class fsmat():
    def __init__(self, Okbar):
        self.Oc = Okbar.T.conj()
        self.O = Okbar 
    def __matmul__(self, x):
        return self.Oc @ self.O @ x + 1e-4*x

class T():
    def __init__(self, f, fav, wf, n_block, diag_reg=1e-4):
        # f: the function used to define the variational derivatives on
        self.f = f
        self.fav = fav
        self.wf = wf
        self.n_block = n_block
        self.diag_reg = diag_reg

    def __matmul__(self, v):
        _, vjp = torch.func.vjp(self.f, self.wf.get_params())
        _, jvp = torch.func.jvp(vjp, (self.wf.get_params(),), (v,))

        res = vjp(jvp)[0]/self.n_block

        _, vjp = torch.func.vjp(self.fav, self.wf.get_params())
        _, jvp = torch.func.jvp(vjp, (self.wf.get_params(),), (v,))
        return res - vjp(jvp)[0] - self.diag_reg * v

class S():
    def __init__(self, f, fav, wf, n_block, diag_reg=1e-4):
        # f: the function used to define the variational derivatives on
        self.f = f
        self.fav = fav
        self.wf = wf
        self.n_block = n_block
        self.diag_reg = diag_reg

    def __matmul__(self, v):
        _, jvp = torch.func.jvp(self.f, (self.wf.get_params(),), (v,))
        _, vjp = torch.func.vjp(self.f, self.wf.get_params())

        res = vjp(jvp)[0]/self.n_block

        _, jvp = torch.func.jvp(self.fav, (self.wf.get_params(),), (v,))
        _, vjp = torch.func.vjp(self.fav, self.wf.get_params())
        return res - vjp(jvp)[0] - self.diag_reg * v


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

    n_spins = 2**14  # spin sites
    alpha = 1
    n_hidden = int(alpha * n_spins)  # neurons in hidden layer
    n_block = 2**5  # samples / expval
    n_epoch = 2**10  # variational iterations
    g = 1.0  # Zeeman interaction amplitude
    eta = torch.tensor(0.01, device=device, dtype=dtype)  # learning rate

    E_exact = ground_state_energy_per_site(g, n_spins)
    ic(E_exact)

    # wf = JST(n_spins, dtype=dtype, device=device)
    wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(n_epoch, n_block, n_spins, wf.n_param)

    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    block = MCBlock(wf, n_block, local_energy=lambda x, y: local_energy(x, y, J=-1, h=-g))
    # block.warmup(wf, tol=1e-1, verbose=1)
    epochbar = trange(n_epoch)
    dTh = torch.zeros(wf.n_param, device=wf.device, dtype=wf.dtype)
    with Profiler(interval=0.1) as profiler:
        for epoch in epochbar:
            # block.bsample_block(wf, n_res=8)
            block.bsample_block_no_grad(wf, n_res=2)

            Eav = torch.mean(block.EL, dim=0)
            epsbar = (block.EL - Eav) / n_block**0.5

            # Okm = torch.mean(block.OK, dim=0)
            # Okbar = (block.OK - Okm) / n_block**0.5
            # Okm = None  # free memory
            # dTh = 2 * Okbar.conj().T @ epsbar
            # if n_block > wf.n_param:
            #     dTh = SR(Okbar, epsbar, diag_reg=1e-4)
            # else:
            #     dTh = minSR(Okbar, epsbar, diag_reg=1e-4)

            vlogpsi = torch.vmap(wf.logprob, in_dims=(0, None))
            f = lambda *primals: vlogpsi(block.samples, *primals)
            _, vjp = torch.func.vjp(f, wf.get_params())

            fav = lambda *primals: vlogpsi(block.samples, *primals).mean()
            _, vjpav = torch.func.vjp(fav, wf.get_params())


            qmetr = S(f, fav, wf, n_block)

            # Smat = (block.OK.T.conj() @ block.OK)
            # Smat = (Okbar.T.conj() @ Okbar)
            # Smat = Smat + 1e-4*torch.eye(Smat.shape[0], dtype=Smat.dtype, device=Smat.device)

            # Smat = torch.eye(Smat.shape[0], dtype=Smat.dtype, device=Smat.device)

            # ov = torch.randn(wf.n_param, device=device, dtype=dtype)
            # ic(qmetr @ ov - Smat @ ov)
            # ic(torch.allclose(qmetr @ ov, Smat @ ov))

            # fimpl = fsmat(Okbar)

            dThn = vjp(block.EL)[0] / n_block - vjpav(Eav)[0]
            x0 = dTh.clone()
            x = cg(qmetr, dThn, x0, max_iter=1)
            # ic(torch.norm(qmetr @ x - dThn))
            dThn = x.clone()

            wf.update_params(-eta * dThn)
            dTh = dThn

            E_var = torch.conj(epsbar) @ epsbar
            edens = Eav.detach().cpu().item()/n_spins
            Eavs[epoch] = edens
            epochbar.set_description(
                f"E/N: {np.round(edens, decimals=4)}, \u03C3: {np.round(E_var.detach().cpu(), decimals=4)}"
            )
    profiler.write_html('our_profiler.html')
    print(Eavs)

# ic((Eav.detach()/n_spins - E_exact))
