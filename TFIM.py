import torch
from tqdm import trange
from ansatz.RBM import RBM
from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler

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

    n_spins = 400  # spin sites
    n_hidden = 400  # neurons in hidden layer
    n_block = 400  # samples / expval
    n_epoch = 100  # variational iterations
    eta = 0.1  # learning rate

    wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)
    ic(wf.n_param)

    epochbar = trange(n_epoch)
    Eavs = torch.zeros(n_epoch, dtype=dtype)
    E_var = 1.0
    with Profiler(interval=0.1) as profiler:
        block = MCBlock(wf, n_block, local_energy=local_energy, verbose=0)
        for epoch in epochbar:
            block.sample_block(wf, n_block)

            Eav = torch.mean(block.EL, dim=0)
            Okm = torch.mean(block.OK, dim=0)

            Okbar = (block.OK - Okm[None, :]) / n_block**0.5
            Okm = None  # free memory
            epsbar = (block.EL - Eav) / n_block**0.5


            U, S, Vh = torch.linalg.svd(Okbar@Okbar.T.conj(), full_matrices=False)

            Smax = torch.max(torch.abs(S))
            # Compute threshold once
            threshold = min(1e-5, E_var.real) * Smax
            # Use torch.where for faster selection and avoid creating intermediate boolean tensor
            mask = torch.where(torch.abs(S) > threshold)[0]
            # Index directly with mask
            U = U.index_select(1, mask)
            S = S.index_select(0, mask)
            Vh = Vh.index_select(0, mask)

            deltaTheta = (
                - eta
                * Okbar.T.conj()
                @ Vh.T.conj()
                @ torch.diag(1/S).to(dtype)
                @ U.T.conj()
                @ epsbar
                )

            wf.update_params(deltaTheta)

            E_var = torch.conj(epsbar) @ epsbar
            # Eavs[epoch] = Eav  # for some reason, this causes a memory leak...
            eta /= 2 if (epoch % 50 == 0) and epoch > 0 else 1
            epochbar.set_description(
                f"Epoch {epoch}, Average energy {Eav.detach()/n_spins}, Energy variance {E_var.detach()}, eta {eta}"
            )
    profiler.print()
