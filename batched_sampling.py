import torch

from tqdm import trange
from ansatz.RBM import RBM

from vmc.iterator import MCBlock
from icecream import ic
from pyinstrument import Profiler
from vmc.minSR import minSR
from sampler.mcmc import draw_next
from energies.TFIM import local_energy
torch._dynamo.config.capture_dynamic_output_shape_ops = True


@torch.compile(fullgraph=True)
def batched_draw_trial(spin_vectors, nflip=1):
    nflips = nflip * spin_vectors.shape[0]
    spin_vectors_flipped = spin_vectors.detach().clone()
    # Generate all random indices at once
    i0 = torch.randint(spin_vectors.shape[0], (nflips,))
    i1 = torch.randint(spin_vectors.shape[1], (nflips,))
    # Flip all selected spins at once
    spin_vectors_flipped[i0, i1] *= -1
    return spin_vectors_flipped

@torch.compile(fullgraph=True, dynamic=True)
def batched_draw_next(wf, x0, n_flip=1, n_iter=2**4):
    spin_vectors = x0.detach().clone()
    # Generate all random numbers upfront
    rand_nums = torch.rand((x0.shape[0], n_iter), device=x0.device, dtype=x0.dtype)

    for i in range(n_iter):
        next_spin_vectors = batched_draw_trial(spin_vectors, n_flip)
        probratios = wf.probratio_(next_spin_vectors, spin_vectors)
        idxs = torch.where(rand_nums[:, i].real <= (probratios * probratios.conj()).real, 1, 0).to(torch.bool)
        if len(idxs) > 0:
            spin_vectors[idxs, :] = next_spin_vectors[idxs, :]
    return spin_vectors

n_spins, n_samples, n_batch = 128, 2**12, 2**12
n_samples_sequential = n_samples//n_batch
wf = torch.compile(RBM(n_spins, n_spins, device='cuda'))
ic(n_batch, n_spins, n_samples, n_samples_sequential, wf.n_param)

sns = 2.0*torch.randint(0, 2, (n_batch, n_spins), dtype=wf.dtype, device=wf.device) - 1
sns = batched_draw_next(wf, sns, n_iter=2**4)  # warmup
with Profiler() as profiler:
    for n in trange(100):
        for m in range(n_samples_sequential):
            sns = batched_draw_next(wf, sns, n_iter=2**4)
profiler.print()


bdraw_next = torch.compile(torch.vmap(lambda x: draw_next(wf, x, n_flip=1, n_iter=2**4), randomness='different'))
sns = bdraw_next(sns)  # warmup
with Profiler() as profiler:
    for n in trange(100):
        for m in range(n_samples_sequential):
            sns = bdraw_next(sns)
profiler.print()