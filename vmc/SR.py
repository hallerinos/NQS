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
    
    return deltaTheta

def SR(Okbar, epsbar, eta, thresh=1e-12):
    Spinv = torch.linalg.pinv(Okbar.T.conj()@Okbar, rtol=thresh)

    deltaTheta = (
        - eta
        * Spinv
        @ Okbar.T.conj()
        @ epsbar
        )
    
    return deltaTheta