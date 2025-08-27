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

    def sample_block(self, wf, n_block, verbose=0, n_dismiss=16):
        spin_vector = 2 * torch.randint(2, [wf.n_spins], device=wf.device, dtype=wf.dtype) - 1.0
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
            # self.OK[n, :] = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))
            # continue
            # check = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))

            wf.reset_gattr()  # reset gradients before calling backward
            wf.logprob(spin_vector).real.backward()
            wf.assign_gradients()
            self.OK[n, :] = wf.gradients.conj()
            # ic(torch.norm(self.OK[n, :] - check))

    def batch_sample_block(self, wf, n_block, verbose=0, n_dismiss=10, n_batch=64):
        spin_vectors = 2 * torch.randint(2, [n_batch, wf.n_spins], device=wf.device, dtype=wf.dtype) - 1
        for n in range(n_dismiss):
            spin_vectors = torch.vmap(draw_next, in_dims=(None, 0), randomness='different')(wf, spin_vectors, n_flip=1, n_iter=4)

        if verbose > 0:
            iterator = trange(n_block // n_batch)
        else:
            iterator = range(n_block // n_batch)

        for n in iterator:
            spin_vectors = torch.vmap(draw_next, in_dims=(None, 0), randomness='different')(wf, spin_vectors, n_flip=1, n_iter=4)
            self.samples[n*n_batch:(n+1)*n_batch, :] = spin_vectors
            self.EL[n*n_batch:(n+1)*n_batch] = torch.vmap(self.local_energy, in_dims=(None, 0))(wf, spin_vectors)


            # wf.assign_derivatives(spin_vector)
            # self.OK[n, :] = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))
            # continue
            # check = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))

            # final thing to vectorize in batches!!!
            # continue
            for (idsv, sv) in enumerate(spin_vectors):
                wf.reset_gattr()  # reset gradients before calling backward
                wf.logprob(sv).real.backward()
                wf.assign_gradients()
                self.OK[n*n_batch + idsv, :] = wf.gradients.conj()
            # ic(torch.norm(self.OK[n, :] - check))
