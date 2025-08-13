import torch
from tqdm import trange
from icecream import ic
from itertools import product


class ExactBlock:
    def __init__(self, wf, verbose=0, local_energy=lambda x, y: 1):
        self.OK = torch.zeros(2**wf.n_spins, wf.n_param, dtype=wf.dtype, device=wf.device)
        self.samples = torch.zeros(
            2**wf.n_spins, wf.n_spins, dtype=wf.dtype, device=wf.device
        )
        self.probs = torch.zeros(2**wf.n_spins, dtype=wf.dtype, device=wf.device)
        self.EL = torch.zeros(2**wf.n_spins, dtype=wf.dtype, device=wf.device)
        self.local_energy = local_energy
        # self.configurations = list(product([-1, 1], repeat=wf.n_spins))
        self.configurations = torch.tensor(list(product([-1, 1], repeat=wf.n_spins)), dtype=wf.dtype, device=wf.device)
        self.compute_exact(wf, verbose=verbose)

    def compute_exact(self, wf, verbose=0, n_dismiss=10):
        for n in range(self.configurations.shape[0]):
            spin_vector = self.configurations[n, :]
            wfa = torch.exp(wf.logprob(spin_vector))
            prob = wfa * wfa.conj()
            self.probs[n] = prob
            self.samples[n, :] = spin_vector
            self.EL[n] = self.local_energy(wf, spin_vector)

            # wf.assign_derivatives(spin_vector)
            # self.OK[n, :] = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))

            # continue

            wf.reset_gattr()  # reset gradients before calling backward
            wf.logprob(spin_vector).real.backward()
            wf.assign_gradients()
            # self.OK[n, :] = torch.cat((wf.b.grad, wf.c.grad, wf.W.grad.flatten()))
            self.OK[n, :] = wf.gradients
            # the following lines evaluate d/dz wf(z) for complex z, matching the definitions of Ob, Oc, and OW
            if torch.is_complex(spin_vector):
                wf.reset_gattr()  # reset gradients before calling backward
                wf.logprob(spin_vector).imag.backward()
                self.OK[n, :] -= wf.gradients*1j

                self.OK[n, :] = self.OK[n, :].conj() / 2
                # ic(torch.norm(self.OK[n, :] - check))
        self.probs /= self.probs.sum()
