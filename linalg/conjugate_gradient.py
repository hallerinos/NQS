import torch
from pyinstrument import Profiler
from tqdm import trange
from collections import OrderedDict
from copy import copy, deepcopy
from icecream import ic

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
        residual = fwmm.compute_residual(xi, k)  # residual = norm(S x[0] - dThd)
    return xi, residual


def bicgstab(fwmm, k: OrderedDict, x0: OrderedDict,
             max_iter=int(1e4), tol=1e-6, compute_residual=False):
    
    dtype = next(iter(k.values())).dtype
    device = next(iter(k.values())).device
    # helper functions
    def block_dot(a: OrderedDict, b: OrderedDict):
        """Computes global inner product ⟨a, b⟩ as a single torch scalar."""
        inner = torch.tensor(0.0, dtype=dtype, device=device)
        for key in a.keys():
            inner += torch.vdot(a[key].flatten(), b[key].flatten())
        return inner

    def block_norm(a: OrderedDict):
        """Computes total Euclidean norm over all blocks."""
        norm = torch.tensor(0.0, dtype=dtype, device=device)
        for key in a.keys():
            norm += a[key].norm()**2 
        return torch.sqrt(norm)
    
    
    xi = copy(x0)

    # r = k - A @ x
    r = OrderedDict()
    Axi = fwmm @ xi
    for key in k.keys():
        r[key] = k[key] - Axi[key]
    r_norm = block_norm(r)
    if r_norm.item() <= tol:
        return xi, 0

    r_hat = copy(r)
    
    rho_old = alpha = omega = torch.tensor(1.0, dtype=dtype, device=device)
    v = OrderedDict({key: torch.zeros_like(val) for key, val in k.items()})
    p = OrderedDict({key: torch.zeros_like(val) for key, val in k.items()})
    
    eps = torch.finfo(next(iter(k.values())).dtype).eps

    for iteration in range(max_iter):

        rho_new = block_dot(r_hat, r)
        if (torch.abs(rho_new)).item() < eps:
            return xi, -1

        if iteration == 0:
            for key in k.keys():
                p[key] = r[key].clone()
        else:
            if (torch.abs(omega)).item() < eps:
                return xi, -2  
            beta = (rho_new / rho_old) * (alpha / omega)
            # p = r + beta * (p - omega * v)
            for key in k.keys():
                p[key] = r[key] + beta * (p[key] - omega * v[key])

        # phat = p # artifact of preconditioning, so far no need for phat
        v = fwmm @ p
        denom = block_dot(r_hat, v)
        if (torch.abs(denom)).item() < eps:
            return xi, -3
        alpha = rho_new / denom

        # if torch.isnan(rho_new) or torch.isnan(alpha) or torch.isnan(omega):
        #     return x, "NaN encountered"

        # s = r - alpha * v
        s = OrderedDict({key: r[key] - alpha * v[key] for key in k.keys()})
        s_norm = block_norm(s)
        if s_norm.item() <= tol:
            # x += alpha * p
            for key in k.keys():
                xi[key] = xi[key] + alpha * p[key]         
            return xi, 0

        t = fwmm @ s
        denom_t = block_dot(t, t)
        if (torch.abs(denom_t)).item() < eps:
            return xi, -4
        omega = block_dot(t, s) / denom_t

        # x and s updates
        for key in k.keys():
            xi[key] = xi[key] + alpha * p[key] + omega * s[key]
            r[key] = s[key] - omega * t[key]

        r_norm = block_norm(r)
        if r_norm.item() <= tol:
            return xi, 0

        rho_old = rho_new.clone()

        # if compute_residual    #TODO

    return xi, 1


if __name__ == "__main__":
    class Adict():
        def __init__(self, Adict):
            self.dict = Adict

        def __matmul__(self, x):
            res = OrderedDict()
            for key in x.keys():
                res[key] = self.dict[key] @ x[key]
            return res
    m, n, dtype, device = int(1111), int(1111), torch.double, 'cpu'

    keys = '12'
    num_iters = 1000
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

    x = cg(amatd, b, x0, max_iter=num_iters)
    ic('CG done')
    res = amatd @ x[0]
    for key in b.keys():
        residuals = res[key] - b[key]
        ic(key)
        ic(residuals)
        ic(torch.norm(residuals))

    
    x = bicgstab(amatd, b, x0, max_iter=num_iters)
    ic('BiCGStab done')
    res = amatd @ x[0]
    for key in b.keys():
        residuals = res[key] - b[key]
        ic(key)
        ic(residuals)
        ic(torch.norm(residuals))