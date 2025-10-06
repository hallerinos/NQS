import torch
from pyinstrument import Profiler
from tqdm import trange
from collections import OrderedDict
from copy import copy, deepcopy

# @torch.compile()
def cg(fwmm, k: torch.tensor, x0: torch.tensor, max_iter=int(1e4), tol=1e-18):
    keys = k.keys()

    xi = deepcopy(x0)

    axi = fwmm @ xi

    pi = OrderedDict()
    for key in keys:
        pi[key] = k[key] - axi[key]

    ri = deepcopy(pi)

    for i in range(1, max_iter):
        api = fwmm @ pi

        rinsq = torch.tensor(0.0)
        for key in keys:
            rinsq.add_(ri[key].flatten().conj() @ ri[key].flatten())

        ai = rinsq
        for key in keys:
            ai.div_(pi[key].flatten().conj() @ api[key].flatten())

        for key in keys:
            xi[key].add_(pi[key].mul_(ai))

        rip = deepcopy(ri)
        for key in keys:
            ri[key].add_(-api[key].mul_(ai))

        rinsqp = rinsq.clone()
        rinsq = torch.tensor(0.0)
        for key in keys:
            rinsq.add_(ri[key].flatten().conj() @ ri[key].flatten())

        if abs(rinsq.item()) < tol:
            return xi, "tol"

        bi = rinsq / rinsqp

        for key in keys:
            pi[key] = ri[key] + pi[key].mul_(bi)
    return xi, "max_iter"

def bicgstab(A, b: torch.tensor, x0=None, *,
             rtol=1e-6, atol=1e-6, max_iter=int(1e4)):
    """
    BiCGSTAB solver for Ax = b.

    Parameters
    ----------
    A : torch.Tensor
        Left-hand side square matrix
    b : torch.Tensor
        Right-hand side vector.
    x0 : torch.Tensor, optional
        Initial guess for solution (default: zero vector).
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    max_iter : int, optional
        Maximum iterations (default: 10 * len(b)).
    M : callable, optional
        Preconditioner operator: y = M(x).
    callback : callable, optional
        Called after each iteration with current x.

    Returns
    -------
    x : torch.Tensor
        Approximate solution to Ax = b.
    info : int
        0 = success, !0 = breakdown or max_iter reached.
    """

    
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    # possible preconditioner
    M = lambda v: v  

    r = b - A @ x
    bnrm2 = b.norm()
    atol = float(atol)
    rtol = float(rtol)
    tol2 = max(atol, rtol * bnrm2) ** 2  # work with squared norms

    if r.norm().pow(2) <= tol2:
        return x, 0

    r_hat = r.clone()
    rho_old = alpha = omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)
    s = torch.empty_like(b)
    t = torch.empty_like(b)

    if max_iter is None:
        max_iter = len(b) * 100

    rhotol = torch.finfo(b.dtype).eps ** 2
    omegatol = rhotol

    for iteration in range(max_iter):

        rho_new = torch.dot(r_hat.conj(), r)
        if torch.abs(rho_new) < rhotol:
            return x, "Breakdown1"

        if iteration == 0:
            p.copy_(r)
        else:
            if torch.abs(omega) < omegatol:
                return x, "Breakdown2"  
            beta = (rho_new / rho_old) * (alpha / omega)
            # p = r + beta * (p - omega * v)
            p.add_(v, alpha=-omega)      # p -= omega * v
            p.mul_(beta)                 # p *= beta
            p.add_(r)                    # p += r

        phat = M(p)
        v.copy_(A @ phat)
        denom = torch.dot(r_hat.conj(), v)
        if torch.abs(denom) < rhotol:
            return x, "Breakdown3"
        alpha = rho_new / denom

        if torch.isnan(rho_new) or torch.isnan(alpha) or torch.isnan(omega):
            return x, "NaN encountered"

        torch.add(r, v, alpha=-alpha, out=s)  # s = r - alpha * v
        if s.norm().pow(2) <= tol2:
            x.add_(phat, alpha=alpha)         # x += alpha * phat
            return x, "0"

        shat = M(s)
        t.copy_(A @ shat)
        denom_t = torch.dot(t.conj(), t)
        if torch.abs(denom_t) < rhotol:
            return x, "Breakdown4"
        omega = torch.dot(t.conj(), s) / denom_t

        x.add_(phat, alpha=alpha)             # x += alpha * phat
        x.add_(shat, alpha=omega)             # x += omega * shat
        torch.add(s, t, alpha=-omega, out=r)  # r = s - omega * t

        if r.norm().pow(2) <= tol2:
            return x, "0"

        rho_old = rho_new

    return x, "max_iter"


if __name__ == "__main__":    
    m, n, dtype = int(111), int(111), torch.double

    keys = '123'
    Adict = OrderedDict()
    for key in keys:
        Ok = torch.randn((m, n), dtype=dtype)
        Ok = torch.where(abs(Ok) < 2, 0, Ok)
        Amat = (Ok.T.conj() @ Ok)
        Amat = Amat + 1e-8*torch.eye(Amat.shape[0], dtype=Ok.dtype)
        Adict[key] = Amat

    def Adict():
        def __init__(Adict):
            self.dict = Adict

        def __matmul__(x):
            res = OrderedDict()
            for key in x.keys():
                res[key] = self.dict[key] @ x[key]
            return res