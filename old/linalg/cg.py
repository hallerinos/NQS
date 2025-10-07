import torch
from pyinstrument import Profiler
from tqdm import trange

# @torch.compile()
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
        if abs(rinsq.item()) < tol:
            return xi, "tol"
        bi = rinsq.item() / rinsqp.item()
        pi = ri + bi * pi
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
    Ok = torch.randn((m, n), dtype=dtype)
    Ok = torch.where(abs(Ok) < 2, 0, Ok)
    Amat = (Ok.T.conj() @ Ok)
    Amat = Amat + 1e-8*torch.eye(Amat.shape[0], dtype=Ok.dtype)
    # Amat = Amat + Amat.T.conj()
    A = lambda x: Amat @ x
    b = torch.randn((Amat.shape[1]), dtype=Amat.dtype)

    # print('Using torch.linalg.solve')
    # bar1 = trange(4)
    # with Profiler(interval=0.1) as profiler:
    #     for _ in bar1:
    #         x0 = torch.randn((Amat.shape[1]), dtype=Amat.dtype)
    #         x = torch.linalg.solve(Amat.to_dense(), b.to_dense())
    #         bar1.set_description(
    #             f"res: {torch.norm(Amat.matmul(x) - b)}"
    #         )

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
            x0 = torch.randn((Amat.shape[1]), dtype=Amat.dtype)
            x = cg(Amat, b, x0)
            bar3.set_description(
                f"res: {torch.norm(Amat @ x - b)}"
            )

   

    print('Using BiCGSTAB: A is a torch tensor')
    bar4 = trange(4)
    with Profiler(interval=0.1) as profiler:
        for _ in bar4:
            x0 = torch.randn((Amat.shape[1]), dtype=Amat.dtype)
            x, info = bicgstab(Amat, b, x0)
            bar4.set_description(
                f"res: {torch.norm(Amat @ x - b)}"
            )
        print(info)


    


    