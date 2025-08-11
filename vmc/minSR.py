import torch

def minSR(Okbar, epsbar, eta, thresh=1e-5):
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
    
    return deltaTheta