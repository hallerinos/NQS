import torch
from collections import OrderedDict

class S():
    def __init__(self, f, fav, model, Ns, diag_reg=0.0):
        # f: the function used to define the variational derivatives on
        self.f = f
        self.fav = fav
        self.model = model
        self.Ns = Ns
        self.diag_reg = diag_reg

    def __matmul__(self, v):
        _, jvp = torch.func.jvp(self.f, (self.model.state_dict(),), (v,))
        _, vjp = torch.func.vjp(self.f, self.model.state_dict())

        res = vjp(jvp)[0]

        _, jvp = torch.func.jvp(self.fav, (self.model.state_dict(),), (v,))
        _, vjp = torch.func.vjp(self.fav, self.model.state_dict())

        res2 = vjp(jvp)[0]

        for key in res.keys():
            res[key].div_(self.Ns)
            res[key].add_(-res2[key])
            res[key].add_(self.diag_reg * v[key])

        return res

    def compute_residual(self, x, rhs):
        residual = 0.0
        for key in x.keys():
            nk = self.__matmul__(x)[key] - rhs[key]
            residual += nk.flatten().conj() @ nk.flatten()
        residual = residual**(0.5)
        return residual