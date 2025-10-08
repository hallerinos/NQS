import torch
from pyinstrument import Profiler
from tqdm import trange
from collections import OrderedDict
from copy import copy, deepcopy

# @torch.compile()
def cg(fwmm, k: OrderedDict, x0: OrderedDict, max_iter=int(1e4), tol=1e-18, compute_residual=False):
    normk = 0.0
    for key in k.keys():
        normk += k[key].norm()
    crit = tol*normk
    xi = copy(x0)
    axi = fwmm @ xi
    pi = OrderedDict()
    for key in k.keys():
        pi[key] = k[key] - axi[key]
    ri = copy(pi)
    for i in range(1, max_iter):
        api = fwmm @ pi
        rinsq = 0
        for key in k.keys():
            rinsq += ri[key].flatten().conj() @ ri[key].flatten()
        ai = 0
        for key in k.keys():
            ai += pi[key].flatten().conj() @ api[key].flatten()
        ai = rinsq / ai
        for key in k.keys():
            xi[key] = xi[key] + ai * pi[key]
        for key in k.keys():
            ri[key] = ri[key] - ai * api[key]
        rinsqp = rinsq.clone()
        rinsq = 0
        for key in k.keys():
            rinsq += ri[key].flatten().conj() @ ri[key].flatten()
        if abs(rinsq.item()) < crit:
            return xi, "tol"
        bi = rinsq / rinsqp
        for key in k.keys():
            pi[key] = ri[key] + bi * pi[key]
    residual = 0
    if compute_residual:
        residual = fwmm.compute_residual(x, k)  # residual = norm(S x[0] - dThd)
    return xi, residual

if __name__ == "__main__":
    class Adict():
        def __init__(self, Adict):
            self.dict = Adict

        def __matmul__(self, x):
            res = OrderedDict()
            for key in x.keys():
                res[key] = self.dict[key] @ x[key]
            return res
    m, n, dtype, device = int(1111), int(1111), torch.double, 'cuda'

    keys = '12'
    ad = OrderedDict()
    b = OrderedDict()
    x0 = OrderedDict()
    for key in keys:
        Ok = torch.randn((m, n), dtype=dtype, device=device)
        Ok = torch.where(abs(Ok) < 2, 0, Ok)
        Amat = (Ok.T.conj() @ Ok)
        Amat = Amat + 1e-8*torch.eye(Amat.shape[0], dtype=Ok.dtype, device=device)
        ad[key] = Amat

        b[key] = torch.randn((m,), dtype=dtype, device=device)
        x0[key] = torch.randn((m,), dtype=dtype, device=device)

    amatd = Adict(ad)

    x = cg(amatd, b, x0)
    res = amatd@x[0]
    for key in b.keys():
        print(res[key] - b[key])
