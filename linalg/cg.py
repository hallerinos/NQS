import torch
from pyinstrument import Profiler
from tqdm import trange

def cg(afun, k: torch.tensor, x0: torch.tensor, max_iter=int(1e4), tol=1e-12):
    xi = x0.clone()
    axi = afun(xi)
    pi = k - axi
    ri = pi.clone()
    for i in range(1, max_iter):
        api = afun(pi)
        rinsq = ri.conj() @ ri
        ai = rinsq / (pi.conj() @ api)
        xi = xi + ai * pi
        rip = ri.clone()
        ri = ri - ai * api
        rinsqp = rinsq.clone()
        rinsq = ri.conj() @ ri
        if rinsq < tol:
            return xi
        bi = rinsq / rinsqp
        pi = ri + bi * pi
    return xi

if __name__ == "__main__":    
    m, n, dtype = int(333), int(3333), torch.double
    Ok = torch.randn((m, n), dtype=dtype)
    Amat = (Ok.T.conj() @ Ok)
    # Amat = (Ok @ Ok.T.conj())

    Amat = Amat + 1e-4*torch.eye(Amat.shape[0], dtype=Ok.dtype)
    # Amat = Amat + Amat.T.conj()
    print(Amat.shape)
    b = torch.randn((Amat.shape[1],), dtype=Amat.dtype)

    bar1 = trange(100)
    with Profiler(interval=0.1) as profiler:
        for _ in bar1:
            x0 = torch.randn((Amat.shape[1],), dtype=Amat.dtype)
            x = torch.linalg.solve(Amat, b)
            bar1.set_description(
                f"res: {torch.norm(Amat.matmul(x) - b)}"
            )

    print('Starting CG using matrix-vector multiplication')
    bar2 = trange(100)
    with Profiler(interval=0.1) as profiler:
        for _ in bar2:
            x0 = torch.randn((Amat.shape[1],), dtype=Amat.dtype)
            x = cg(Amat.matmul, b, x0)
            bar2.set_description(
                f"res: {torch.norm(Amat.matmul(x) - b)}"
            )


    print('Starting CG using regularized vec-vec multiplication')
    bar3 = trange(100)
    myfun = lambda vec: Ok.T.conj() @ (Ok @ vec) + 1e-4*vec
    with Profiler(interval=0.1) as profiler:
        for _ in bar3:
            x0 = torch.randn((Amat.shape[1],), dtype=Amat.dtype)
            x = cg(myfun, b, x0)
            bar3.set_description(
                f"res: {torch.norm(myfun(x) - b)}"
            )