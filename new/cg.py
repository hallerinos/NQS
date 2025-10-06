import torch
from pyinstrument import Profiler
from tqdm import trange
from collections import OrderedDict
from copy import copy, deepcopy

# @torch.compile()
def cg(fwmm, k: OrderedDict, x0: OrderedDict, max_iter=int(1e4), tol=1e-18):
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
        if abs(rinsq.item()) < tol:
            return xi, "tol"
        bi = rinsq / rinsqp
        for key in k.keys():
            pi[key] = ri[key] + bi * pi[key]
    return xi, "max_iter"

class Adict():
    def __init__(self, Adict):
        self.dict = Adict

    def __matmul__(self, x):
        res = OrderedDict()
        for key in x.keys():
            res[key] = self.dict[key] @ x[key]
        return res

if __name__ == "__main__":    
    m, n, dtype = int(111), int(111), torch.double

    keys = '12'
    ad = OrderedDict()
    b = OrderedDict()
    x0 = OrderedDict()
    for key in keys:
        Ok = torch.randn((m, n), dtype=dtype)
        Ok = torch.where(abs(Ok) < 2, 0, Ok)
        Amat = (Ok.T.conj() @ Ok)
        Amat = Amat + 1e-8*torch.eye(Amat.shape[0], dtype=Ok.dtype)
        ad[key] = Amat

        b[key] = torch.randn((m,), dtype=dtype)
        x0[key] = torch.randn((m,), dtype=dtype)

    amatd = Adict(ad)

    x = cg(amatd, b, x0)
    res = amatd@x[0]
    for key in b.keys():
        print(res[key] - b[key])
