import torch
from sampler.mcmc import draw_next
from tqdm import trange
from icecream import ic
from ansatz.RBM import derivatives, autograd
import torch._dynamo
torch._dynamo.config.suppress_errors = True


# @torch.compile(fullgraph=False)
class MCBlock:
    def __init__(self, wf, n_block, local_energy=lambda x, y: 1):
        # self.OK = torch.zeros(n_block, wf.n_param, dtype=wf.dtype, device=wf.device)
        self.samples = (2.0*torch.randint(0, 2, (n_block, wf.n_spins), device=wf.device) - 1.0).to(wf.dtype)
        self.EL = torch.zeros(n_block, dtype=wf.dtype, device=wf.device)
        self.local_energy = local_energy

    # @torch.compile(fullgraph=False)
    def sample_block(self, wf, n_block, verbose=0, n_discard=2**4, n_iter=2**4):
        spin_vector = 2 * torch.randint(2, [wf.n_spins], device=wf.device, dtype=wf.dtype) - 1.0
        for n in range(n_discard):
            spin_vector = draw_next(wf, spin_vector, n_flip=1, n_iter=4)

        if verbose > 0:
            iterator = trange(n_block)
        else:
            iterator = range(n_block)

        for n in iterator:
            spin_vector = draw_next(wf, spin_vector, n_flip=1, n_iter=4)
            self.samples[n, :] = spin_vector
            self.EL[n] = self.local_energy(wf, spin_vector)


            wf.assign_derivatives(spin_vector)
            self.OK[n, :] = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))
            continue
            # check = torch.cat((wf.Ob, wf.Oc, wf.OW.flatten()))

            wf.reset_gattr()  # reset gradients before calling backward
            wf.logprob(spin_vector).real.backward()
            wf.assign_gradients()
            self.OK[n, :] = wf.gradients.conj()
            # ic(torch.norm(self.OK[n, :] - check))

    # @torch.compile(fullgraph=False)
    def bsample_block(self, wf, n_res=4):
        # bdraw_next = torch.compile(torch.vmap(lambda x: draw_next(wf, x, n_flip=1, n_iter=2**4), randomness='different'))
        # blocal_energy = torch.compile(torch.vmap(lambda x: self.local_energy(wf, x)))
        # # vmgrad = torch.compile(torch.vmap(lambda x: derivatives(wf, x)))
        # vmagrad = torch.compile(torch.vmap(lambda x: autograd(wf, x)))

        bdraw_next = torch.vmap(lambda x: draw_next(wf, x, n_flip=1, n_iter=2**4), randomness='different')
        blocal_energy = torch.vmap(lambda x: self.local_energy(wf, x))
        # vmgrad = torch.vmap(lambda x: derivatives(wf, x))
        vmagrad = torch.vmap(lambda x: autograd(wf, x))

        for _ in range(n_res):
            self.samples = bdraw_next(self.samples)
        self.EL = blocal_energy(self.samples)

        self.OK = vmagrad(self.samples)
        # self.OK = wf.gradients

    # @torch.compile(fullgraph=False)
    def bsample_block_no_grad(self, wf, n_res=4):
        # bdraw_next = torch.compile(torch.vmap(lambda x: draw_next(wf, x, n_flip=1, n_iter=2**4), randomness='different'))
        # blocal_energy = torch.compile(torch.vmap(lambda x: self.local_energy(wf, x)))
        # # vmgrad = torch.compile(torch.vmap(lambda x: derivatives(wf, x)))
        # vmagrad = torch.compile(torch.vmap(lambda x: autograd(wf, x)))

        bdraw_next = torch.vmap(lambda x: draw_next(wf, x, n_flip=1, n_iter=1), randomness='different')
        blocal_energy = torch.vmap(lambda x: self.local_energy(wf, x))
        # vmgrad = torch.vmap(lambda x: derivatives(wf, x))
        # vmagrad = torch.vmap(lambda x: autograd(wf, x))

        for _ in range(n_res):
            self.samples = bdraw_next(self.samples)
        self.EL = blocal_energy(self.samples)

        # self.OK = vmagrad(self.samples)
        # self.OK = wf.gradients

    # @torch.compile(fullgraph=False)
    def warmup(self, wf, tol=1e-2, verbose=False):
        # bdraw_next = torch.compile(torch.vmap(lambda x: draw_next(wf, x, n_iter=2**4), randomness='different'))
        # blocal_energy = torch.compile(torch.vmap(lambda x: self.local_energy(wf, x)))
        bdraw_next = torch.vmap(lambda x: draw_next(wf, x, n_iter=2**4), randomness='different')
        blocal_energy = torch.vmap(lambda x: self.local_energy(wf, x))

        pEav = torch.mean(self.EL)
        Eav = torch.mean(self.EL)
        crit = torch.ones_like(pEav).item()
        while crit > tol:
            pEav = Eav
            self.samples = bdraw_next(self.samples)
            self.EL = blocal_energy(self.samples)
            Eav = torch.mean(self.EL)
            crit = torch.abs(pEav - Eav).item()
            if verbose:
                print(crit)