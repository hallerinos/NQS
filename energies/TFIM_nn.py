import torch
from ansatz.RBM import RBM

# @torch.compile(fullgraph=False)
def local_energy(wf: RBM, spin_vector: torch.Tensor, J=-1, h=-1):
    interactions = torch.einsum('ij,ij->i', spin_vector[:, 1:], spin_vector[:,:-1])
    interactions += torch.einsum('ij,ij->i', spin_vector[:,[0]], spin_vector[:,[-1]].T)

    zeeman_term = torch.zeros_like(interactions)
    p = wf(spin_vector)
    for i in range(spin_vector.shape[1]):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[:,i] *= -1
        p_flipped = wf(spin_vector_f)
        zeeman_term.add_(torch.exp(p_flipped - p))

    return J * interactions + h * zeeman_term
