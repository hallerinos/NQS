import torch
from ansatz.RBM import RBM
from tqdm import trange
from vmc.iterator import MCBlock


def local_energy(wf: RBM, spin_vector: torch.Tensor, J=-1.0, h=-1.0):
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
    n_epoch = 1000  # variational iterations
    eta = 0.1  # learning rate

    wf = RBM(n_spins, n_hidden, dtype=dtype, device=device)

    epochbar = trange(n_epoch)
    Eavs = torch.zeros(n_epoch, dtype=dtype)
    for epoch in epochbar:
        block = MCBlock(wf, n_block, local_energy=local_energy, verbose=0)

        Eav = torch.mean(block.EL, dim=0)
        Okm = torch.mean(block.OK, dim=0)

        Okbar = (block.OK - Okm[None, :]) / n_block**0.5
        Okm = None  # free memory
        epsbar = (block.EL - Eav) / n_block**0.5
        T = Okbar @ Okbar.adjoint()

        deltaTheta = (
            -eta
            * Okbar.adjoint()
            @ torch.pinverse(
                T + 1e-6 * torch.eye(T.shape[0], dtype=T.dtype, device=T.device),
                rcond=1e-6,
            )
            @ epsbar
        )

        wf.update_params(deltaTheta)

        E_var = torch.conj(epsbar) @ epsbar
        Eavs[epoch] = Eav
        eta /= 2 if (epoch % 50 == 0) and epoch > 0 else 1
        epochbar.set_description(
            f"Epoch {epoch}, Average energy {Eav.detach()}, Energy variance {E_var.detach()}, eta {eta}"
        )
