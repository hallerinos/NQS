import torch
from sampler.mcmc import draw_next
from tqdm import trange
from icecream import ic


class MCBlock:
    def __init__(self, wf, n_block, verbose=0, local_energy=lambda x, y: 1):
        self.OK = torch.zeros(n_block, wf.n_param, dtype=wf.dtype, device=wf.device)
        self.samples = torch.zeros(
            n_block, wf.n_spins, dtype=wf.dtype, device=wf.device
        )
        self.EL = torch.zeros(n_block, dtype=wf.dtype, device=wf.device)
        self.local_energy = local_energy
        self.sample_block(wf, n_block, verbose)

    def sample_block(self, wf, n_block, verbose=0, n_dismiss=10):
        spin_vector = 2 * torch.randint(2, [wf.n_spins], device=wf.device) - 1
        spin_vector = spin_vector.to(wf.dtype)
        for n in range(n_dismiss):
            spin_vector = draw_next(wf, spin_vector, n_flip=1, n_iter=4)

        if verbose > 0:
            iterator = trange(n_block)
        else:
            iterator = range(n_block)

        for n in iterator:
            spin_vector = draw_next(wf, spin_vector, n_flip=1, n_iter=4)
            self.samples[n, :] = spin_vector
            self.EL[n] = self.local_energy(wf, spin_vector)


            # wf.assign_derivatives(spin_vector)
            # check = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))

            wf.reset_gattr()  # reset gradients before calling backward
            wf.logprob(spin_vector).real.backward()
            wf.assign_gradients()
            self.OK[n, :] = wf.gradients.conj()
            # ic(torch.norm(self.OK[n, :] - check))
