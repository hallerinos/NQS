import torch
from collections import OrderedDict

class S():
    def __init__(self, f, fav, model, Ns, diag_reg=1e-4):
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

        for (key, val) in res.items():
            res[key].div_(self.Ns)
            res[key].add_(-res2[key])
            res[key].add_(-self.diag_reg * v[key])

        # resvals = torch.cat([val.flatten() for val in resdict.values()])

        return res