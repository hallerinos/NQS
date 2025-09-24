import torch
from pyinstrument import Profiler
from tqdm import trange

# @torch.compile(dynamic=True, options={"trace.enabled":False})
def cg(fwmm, k: torch.tensor, x0: torch.tensor, max_iter=int(1e4), tol=1e-18):
    xi = x0.clone()
    axi = fwmm @ xi
    pi = k - axi
    ri = pi.clone()
    for i in range(1, max_iter):
        api = fwmm @ pi
        rinsq = ri.conj() @ ri
        ai = rinsq / (pi.conj() @ api).item()
        xi = xi + ai.item() * pi
        rip = ri.clone()
        ri = ri - ai.item() * api
        rinsqp = rinsq.clone()
        rinsq = ri.conj() @ ri
        if rinsq.item() < tol:
            return xi
        bi = rinsq.item() / rinsqp.item()
        pi = ri + bi * pi
    return xi

if __name__ == "__main__":    
    m, n, dtype = int(111), int(1111), torch.double
    Ok = torch.randn((m, n), dtype=dtype)
    Ok = torch.where(abs(Ok) < 2, 0, Ok)
    # print(Ok)
    Amat = (Ok.T.conj() @ Ok)
    Amat = Amat + 1e-8*torch.eye(Amat.shape[0], dtype=Ok.dtype)
    # Amat = Amat + Amat.T.conj()
    b = torch.randn((Amat.shape[1],1), dtype=Amat.dtype)

    print('Using torch.linalg.solve')
    bar1 = trange(4)
    with Profiler(interval=0.1) as profiler:
        for _ in bar1:
            x0 = torch.randn((Amat.shape[1],), dtype=Amat.dtype)
            x = torch.linalg.solve(Amat.to_dense(), b.to_dense())
            bar1.set_description(
                f"res: {torch.norm(Amat.matmul(x) - b)}"
            )

    # print('Starting CG using matrix-vector multiplication')
    # bar2 = trange(4)
    # with Profiler(interval=0.1) as profiler:
    #     for _ in bar2:
    #         x0 = torch.randn((Amat.shape[1], 1), dtype=Amat.dtype).to_sparse()
    #         x = cg(Amat.mm, b, x0)
    #         bar2.set_description(
    #             f"res: {torch.norm(Amat.matmul(x) - b)}"
    #         )


    print('Starting CG using regularized vec-vec multiplication')
    bar3 = trange(4)
    with Profiler(interval=0.1) as profiler:
        for _ in bar3:
            x0 = torch.randn((Amat.shape[1],1), dtype=Amat.dtype)
            x = cg(Amat, b, x0)
            bar3.set_description(
                f"res: {torch.norm(Amat @ x - b)}"
            )