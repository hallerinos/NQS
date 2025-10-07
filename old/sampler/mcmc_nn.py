import torch

# @torch.compile(fullgraph=False)
def draw_trial(spin_vector, nflip):
    spin_vector_flipped = spin_vector.detach().clone()
    # Generate all random indices at once
    indices = torch.randint(len(spin_vector), (nflip,))
    # Flip all selected spins at once
    spin_vector_flipped[indices] *= -1
    return spin_vector_flipped

# compute the cross-correlation between two samples
def rho(sample1, sample2):
    avsample1 = torch.mean(sample1, dim=0)
    avsample2 = torch.mean(sample2, dim=0)
    covod = torch.mean((sample1 - avsample1) * (sample2 - avsample2), dim=0) / (sample1.shape[0] - 1)
    covd1 = torch.mean((sample1 - avsample1)**2, dim=0) / (sample1.shape[0] - 1)
    covd2 = torch.mean((sample2 - avsample2)**2, dim=0) / (sample2.shape[0] - 1)
    return covod / torch.sqrt(covd1 * covd2), covod, covd1, covd2

# @torch.compile(fullgraph=False)
def draw_next(wf, x0, n_flip=1, n_iter=10):
    spin_vector = x0.detach().clone()
    # Generate all random numbers upfront
    rand_nums = torch.rand(n_iter, device=x0.device, dtype=x0.dtype)

    # consider optimizing the sampling -- how to make it parallel?
    for i in range(n_iter):
        next_spin_vector = draw_trial(spin_vector, n_flip)
        p_next = wf(next_spin_vector)
        p = wf(spin_vector)
        probratio = torch.exp(p_next - p)
        # probratio = wf.probratio(next_spin_vector, spin_vector)
        spin_vector = torch.where(rand_nums[i].real <= (probratio * probratio.conj()).real, next_spin_vector, spin_vector)
    return spin_vector