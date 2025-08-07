import torch
from tqdm import trange
from ansatz.RBM import RBM
from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler

@torch.compile(fullgraph=True)
def local_energy(wf: RBM, spin_vector: torch.Tensor, J=-0.5, h=-1):
    interactions = spin_vector[1:] @ spin_vector[:-1]
    interactions += spin_vector[0] * spin_vector[-1]  # pbc

    zeeman_term = 0
    for n in range(len(spin_vector)):
        spin_vector_flipped = spin_vector.detach().clone()
        spin_vector_flipped[n] *= -1
        zeeman_term += wf.probratio(spin_vector_flipped, spin_vector)

    return J * interactions + h * zeeman_term


if __name__ == "__main__":
    device = "cpu"
    dtype = torch.double
    # dtype = torch.complex128

    print(torch.__version__)

    n_spins = 10  # spin sites
    n_hidden = 10  # neurons in hidden layer
    n_block = 1000  # samples / expval
    n_epoch = 10  # variational iterations
    eta = 0.1  # learning rate

    wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(wf.n_param)

    epochbar = trange(n_epoch)
    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    with Profiler(interval=0.1) as profiler:
        for epoch in epochbar:
            block = MCBlock(wf, n_block, local_energy=local_energy, verbose=0)  # this suffers from a memory leak

            Eav = torch.mean(block.EL, dim=0)
            Okm = torch.mean(block.OK, dim=0)

            Okbar = (block.OK - Okm[None, :]) / n_block**0.5
            Okm = None  # free memory
            epsbar = (block.EL - Eav) / n_block**0.5

            U, S, Vh = torch.linalg.svd(Okbar, full_matrices=False)

            Smax = max(abs(S))
            sel = abs(S) > min(1e-5, E_var)*Smax
            U = U[:, sel]
            S = S[sel]
            Vh = Vh[sel, :]

            deltaTheta = (
                -eta
                * Vh.T.conj()
                @ torch.diag(1/S)
                @ U.T.conj()
                @ epsbar
                )

            wf.update_params(deltaTheta)

            E_var = torch.conj(epsbar) @ epsbar
            # Eavs[epoch] = Eav  # for some reason, this causes a memory leak...
            eta /= 2 if (epoch % 50 == 0) and epoch > 0 else 1
            epochbar.set_description(
                f"Epoch {epoch}, Average energy {Eav.detach()}, Energy variance {E_var.detach()}, eta {eta}"
            )
    profiler.print()
