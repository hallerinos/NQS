import torch


def SR_(Okbar, epsbar, eta, thresh=1e-12):
    U, S, Vh = torch.linalg.svd(Okbar.T.conj()@Okbar, full_matrices=False)

    Smax = torch.max(torch.abs(S))
    # Compute threshold once
    threshold = thresh * Smax
    # Use torch.where for faster selection and avoid creating intermediate boolean tensor
    mask = torch.where(torch.abs(S) > threshold)[0]
    # Index directly with mask
    U = U.index_select(1, mask)
    S = S.index_select(0, mask)
    Vh = Vh.index_select(0, mask)

    deltaTheta = (
        - eta
        * Vh.T.conj()
        @ torch.diag(1/S).to(Okbar.dtype)
        @ U.T.conj()
        @ Okbar.T.conj()
        @ epsbar
    )
    return 2 * deltaTheta

# @torch.compile(fullgraph=False)


def SR(Okbar, epsbar, diag_reg=1e-3):
    # Spinv = torch.linalg.pinv(Okbar.T.conj()@Okbar, rtol=thresh, hermitian=False)
    with torch.no_grad():
        Okbarc = Okbar.T.conj()
        matrix = Okbarc @ Okbar + diag_reg * \
            torch.eye(Okbar.shape[1], dtype=Okbar.dtype, device=Okbar.device)
        rhs = 2.0 * Okbarc @ epsbar
        # see https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_ex.html#torch.linalg.solve_ex (and without the _ex)
        deltaTheta, info = torch.linalg.solve_ex(matrix, rhs)
    return deltaTheta

# @torch.compile(fullgraph=False)


def SR__(Okbar, epsbar, diag_reg=1e-6, diag_shift=0.01):
    Okbarc = Okbar.T.conj()
    Spinv = torch.linalg.pinv(Okbarc@Okbar + diag_shift*torch.eye(
        Okbar.shape[1], dtype=Okbar.dtype, device=Okbar.device), rtol=diag_reg, hermitian=True)

    deltaTheta = Spinv @ Okbarc @ epsbar
    return 2 * deltaTheta
