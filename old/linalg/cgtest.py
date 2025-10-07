import torch, cusolve
from pyinstrument import Profiler
from tqdm import trange

def cg(x1: torch.tensor, x2: torch.tensor, max_iter=int(1e4), tol=1e-18):
    for i in trange(1, max_iter):
        x1.matmul(x2)
    return

if __name__ == "__main__":
    m, n, dtype = int(2**11), int(2**11), torch.double
    print(m, n)

    in1 = torch.rand((n,m), device='cuda', dtype=torch.double)
    in2 = torch.rand((m,n), device='cuda', dtype=torch.double)

    with Profiler(interval=0.1) as profiler:
        for _ in trange(32):
            cg(in1, in2, max_iter=1000000)
    profiler.print()