def GD(Okbar, epsbar, eta, **kwargs):
    deltaTheta = (
        - eta
        * Okbar.T.conj()
        @ epsbar
        )
    
    return deltaTheta