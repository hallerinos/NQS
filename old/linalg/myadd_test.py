import torch, cusolve
from pyinstrument import Profiler
from tqdm import trange

n, m = 3333, 3333
in1 = torch.rand((n,m), device='cuda')
in2 = torch.rand((m,n), device='cuda')

with Profiler(interval=0.001) as profiler:
    for _ in trange(111):
        x = in1 @ in2
        x = None
profiler.print()

with Profiler(interval=0.001) as profiler:
    for _ in trange(111):
        x = in1 @ in2
        x = None
profiler.print()

with Profiler(interval=0.001) as profiler:
    for _ in trange(111):
        cusolve.mymm(in1, in2, 11)
profiler.print()