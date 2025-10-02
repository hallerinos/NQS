import torch
from ansatz.RBM import RBM

# @torch.compile(fullgraph=False)
def local_energy(wf: RBM, spin_vector: torch.Tensor, J=-1, h=-1):
    interactions = spin_vector[1:] @ spin_vector[:-1]
    interactions += spin_vector[0] * spin_vector[-1]  # pbc

    zeeman_term = torch.zeros_like(interactions)
    # sequential adding energy of spin flips (less memory, more time)
    for i in range(spin_vector.shape[0]):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[i] *= -1
        p_flipped = wf(spin_vector_f)
        p = wf(spin_vector)
        zeeman_term += torch.exp(p_flipped - p)
        # zeeman_term += wf.probratio(spin_vector_f, spin_vector)
        # zeeman_term += wf.probratio(spin_vector_f, spin_vector)
    # # Create copies of original configuration and a batch of flipped configurations
    # spin_vector_r = spin_vector.repeat(len(spin_vector), 1)
    # flipped_spins = spin_vector.repeat(len(spin_vector), 1)
    # # Create indices for diagonal elements
    # indices = torch.arange(len(spin_vector), device=spin_vector.device)
    # # Flip spins in batch
    # flipped_spins[indices, indices] *= -1
    # # Calculate all probability ratios at once
    # zeeman_term = wf.probratio_(flipped_spins, spin_vector_r).sum()

    return J * interactions + h * zeeman_term
