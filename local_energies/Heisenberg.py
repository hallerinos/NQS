import torch

def Heisenberg(model, spin_vector: torch.Tensor, J=[-1.0,-1.0,-1.0], B=[0.0, 0.0, -1.0]) -> torch.Tensor:
    ZZi = torch.einsum('ij,ij->i', spin_vector[:, 1:], spin_vector[:,:-1])
    # ZZi += torch.einsum('ij,ij->i', spin_vector[:,[0]], spin_vector[:,[-1]])

    XYi = torch.zeros_like(ZZi)
    lnwf0 = model(spin_vector)
    for i in range(spin_vector.shape[1]-1):
        spin_vector_f = spin_vector.clone()
        spin_vector_f[:, [i, i+1]] *= -1
        lnwf1 = model(spin_vector_f)
        XYi.add_((J[0] - J[1]*spin_vector[:, i]*spin_vector[:, i+1])*torch.exp(lnwf1 - lnwf0))
    # spin_vector_f = spin_vector.clone()
    # spin_vector_f[:, [0, -1]] *= -1
    # lnwf1 = model(spin_vector_f)
    # XYi.add_((1.0 - spin_vector[:, i]*spin_vector[:, i+1])*torch.exp(lnwf1 - lnwf0))

    # zeeman_z = B[2] * torch.sum(spin_vector, dim=-1)
    # zeeman_xy = torch.zeros_like(zeeman_z)
    # for i in range(spin_vector.shape[1]):
    #     spin_vector_f = spin_vector.clone()
    #     spin_vector_f[:, i] *= -1
    #     lnwf1 = model(spin_vector_f)
    #     val = torch.exp(lnwf1 - lnwf0)
    #     zeeman_xy.add_((B[0] + B[1]*spin_vector[:, i]*1j)*val)

    res = (J[2] * ZZi + XYi)/4

    return res

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
