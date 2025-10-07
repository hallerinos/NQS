import torch

def minSR_(Okbar, epsbar, eta, thresh=1e-12):
    U, S, Vh = torch.linalg.svd(Okbar@Okbar.T.conj(), full_matrices=False)

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
        * Okbar.T.conj()
        @ Vh.T.conj()
        @ torch.diag(1/S).to(Okbar.dtype)
        @ U.T.conj()
        @ epsbar
        )
    
    return 2 * deltaTheta

# @torch.compile(fullgraph=False)
def minSR(Okbar, epsbar, diag_reg=1e-12):
    # Tpinv = torch.linalg.pinv(Okbar@Okbar.T.conj(), rtol=thresh, hermitian=False)
    # Tpinvepsbar = Tpinv @ epsbar
    Okbarc = Okbar.T.conj()
    Tpinvepsbar = torch.linalg.solve(Okbar@Okbarc + diag_reg*torch.eye(Okbar.shape[0], dtype=Okbar.dtype, device=Okbar.device), epsbar)  # see https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_ex.html#torch.linalg.solve_ex (and without the _ex)

    deltaTheta = Okbarc @ Tpinvepsbar

    return 2 * deltaTheta