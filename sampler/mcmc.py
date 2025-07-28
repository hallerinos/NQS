import torch

def draw_trial(spin_vector, nflip):
    spin_vector_flipped = spin_vector.detach().clone()
    for _ in range(nflip):
        n = torch.randint(len(spin_vector), [1]).item()
        spin_vector_flipped[n] *= -1
    return spin_vector_flipped

# compute the cross-correlation between two samples
def rho(sample1, sample2):
    avsample1 = torch.mean(sample1, dim=0)
    avsample2 = torch.mean(sample2, dim=0)
    covod = torch.mean((sample1 - avsample1) * (sample2 - avsample2), dim=0) / (sample1.shape[0] - 1)
    covd1 = torch.mean((sample1 - avsample1)**2, dim=0) / (sample1.shape[0] - 1)
    covd2 = torch.mean((sample2 - avsample2)**2, dim=0) / (sample2.shape[0] - 1)
    return covod / torch.sqrt(covd1 * covd2), covod, covd1, covd2


def draw_next(wf, x0, n_flip=1, n_iter=10):
    spin_vector = x0.detach().clone()
    for _ in range(n_iter):
        next_spin_vector = draw_trial(spin_vector, n_flip)
        p_accept = torch.pow(torch.abs(wf.probratio(next_spin_vector, spin_vector)), 2)
        if torch.rand(1) <= p_accept:
            spin_vector = next_spin_vector
    return spin_vector