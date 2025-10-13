import torch

def TFIM(model, spin_vector: torch.Tensor, J:torch.NumberType=-1, h:torch.NumberType=-1) -> torch.Tensor:
    interactions = torch.einsum('ij,ij->i', spin_vector[:, 1:], spin_vector[:,:-1])
    interactions += torch.einsum('ij,ij->i', spin_vector[:,[0]], spin_vector[:,[-1]])

    zeeman_term = torch.zeros_like(interactions)
    lnwf0 = model(spin_vector)
    for i in range(spin_vector.shape[1]):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[:, i] *= -1
        lnwf1 = model(spin_vector_f)
        zeeman_term.add_(torch.exp(lnwf1 - lnwf0))

    return J * interactions + h * zeeman_term

def TFIM_y(model, spin_vector: torch.Tensor, J:torch.NumberType=-1, h:torch.NumberType=-1) -> torch.Tensor:
    interactions = torch.einsum('ij,ij->i', spin_vector[:, 1:], spin_vector[:,:-1])
    interactions += torch.einsum('ij,ij->i', spin_vector[:,[0]], spin_vector[:,[-1]])

    zeeman_term = torch.zeros_like(interactions)
    lnwf0 = model(spin_vector)
    for i in range(spin_vector.shape[1]):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[:, i] *= -1
        lnwf1 = model(spin_vector_f)
        zeeman_term.add_((-1j*spin_vector[:,i])*torch.exp(lnwf1 - lnwf0))

    return J * interactions + h * zeeman_term

def TFIM_rot(model, spin_vector: torch.Tensor, J:torch.NumberType=-1, h:torch.NumberType=-1) -> torch.Tensor:
    zeeman_term = torch.sum(spin_vector, dim=-1)

    interactions = torch.zeros_like(zeeman_term).to(model.device, model.dtype)
    lnwf0 = model(spin_vector)
    for i in range(spin_vector.shape[1]-1):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[:, [i,i+1]] *= -1.0
        lnwf1 = model(spin_vector_f)
        interactions.add_(torch.exp(lnwf1 - lnwf0))
    spin_vector_f = spin_vector.clone()
    spin_vector_f[:, [-1,0]] *= -1.0
    lnwf1 = model(spin_vector_f)
    interactions.add_(torch.exp(lnwf1 - lnwf0))


    return J * interactions + h * zeeman_term
