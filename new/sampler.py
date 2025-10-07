import torch
from icecream import ic
import tqdm
import numpy as np

class sampler:
    def __init__(self, model, n_block, local_energy=lambda x, y: 1):
        self.model = model
        self.n_block = n_block
        self.samples = (2.0*torch.randint(0, 2, (n_block, model.n_spins)) - 1.0).to(torch.get_default_device(), torch.get_default_dtype())
        self.EL = torch.zeros(n_block).to(torch.get_default_device(), torch.get_default_dtype())
        self.local_energy = local_energy

    # @torch.compile()
    def draw_trial(self, n_flip=1):
        spin_vector_flipped = self.samples.clone()
        # Generate all random indices at once
        indices = torch.randint(self.samples.shape[1], (n_flip,))
        # Flip all selected spins at once
        spin_vector_flipped[:, indices] *= -1
        return spin_vector_flipped

    # draws new configurations, returns acceptance rate
    # @torch.compile()
    def draw_next(self, n_res=1, n_flip=1):
        # create all random numbers at once
        rand_nums = torch.rand((n_res, self.n_block), device=torch.get_default_device(), dtype=torch.get_default_dtype())
        for i in range(n_res):
            y = self.draw_trial(n_flip=n_flip)
            probratio = self.model.probratio(y, self.samples)
            # lnwf0 = self.model(self.samples)
            # lnwf1 = self.model(y)
            # probratio = torch.exp(lnwf1 - lnwf0)
            accepted = (rand_nums[i].real <= (probratio * probratio.conj()).real)
            self.samples[accepted] = y[accepted]
        return accepted.to(torch.int).sum() / self.n_block

    # @torch.compile()
    def warmup(self, n_res=4, n_flip=1, tol=1e-4, min_iter=2**6, max_iter=2**10):
        E_prev = 0
        tbar = tqdm.trange(max_iter)
        for step in tbar:
            nacc = self.draw_next(n_res=n_res, n_flip=n_flip)
            self.EL = self.local_energy(self.model, self.samples)

            E_next = self.EL.mean()
            dE = E_next - E_prev
            E_prev = E_next.clone()
            crit = abs(dE/E_prev)

            Evar = self.EL - E_next
            Evar.mul_(Evar)

            Edens = E_next/len(self.samples[0])
            tbar.set_description(
                f"E/N: {np.round(Edens.cpu(), decimals=4)}, \u03C3: {np.round(Evar.mean().cpu(), decimals=4)}"
            )
            if crit < tol and nacc > 1e-1 and step > min_iter:
                break

    # @torch.compile()
    def sample(self, n_res=4, n_flip=1):
        nacc = self.draw_next(n_res=n_res, n_flip=n_flip)
        self.EL = self.local_energy(self.model, self.samples)