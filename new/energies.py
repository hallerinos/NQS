import torch

# @torch.compile(fullgraph=False)
def TFIM(wf, spin_vector: torch.Tensor, J:torch.NumberType=-1, h:torch.NumberType=-1) -> torch.Tensor:
    interactions = torch.einsum('ij,ij->i', spin_vector[:, 1:], spin_vector[:,:-1])
    interactions += torch.einsum('ij,ij->i', spin_vector[:,[0]], spin_vector[:,[-1]])

    zeeman_term = torch.zeros_like(interactions)
    lnwf0 = wf(spin_vector)
    for i in range(spin_vector.shape[1]):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[:, i] *= -1
        lnwf1 = wf(spin_vector_f)
        zeeman_term.add_(torch.exp(lnwf1 - lnwf0))

    return J * interactions + h * zeeman_term
