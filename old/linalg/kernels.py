import torch

class T():
    def __init__(self, f, fav, wf, n_block, diag_reg=1e-4):
        # f: the function used to define the variational derivatives on
        self.f = f
        self.fav = fav
        self.wf = wf
        self.n_block = n_block
        self.diag_reg = diag_reg

        _, vjp = torch.func.vjp(self.f, self.wf.get_params())
        _, vjpav = torch.func.vjp(self.fav, self.wf.get_params())
        self.vjp = vjp
        self.vjpav = vjpav

        self.jvp = lambda v: torch.func.jvp(self.f, (self.wf.get_params(),), (v,))[1]
        self.jvpav = lambda v: torch.func.jvp(self.fav, (self.wf.get_params(),), (v,))[1]

    def __matmul__(self, v):
        vm = v.mean()

        # 1st term (see our notes)
        t1 = self.jvp(self.vjp(v)[0]) / self.n_block

        # 2st term (see our notes)
        t2 = self.jvp(self.vjpav(vm)[0])

        # 3st term (see our notes)
        t3 = self.jvpav(self.vjp(v)[0]) / self.n_block

        # 4st term (see our notes)
        t4 = self.jvpav(self.vjpav(vm)[0])

        res = t1 - t2 - t3 + t4

        return res + self.diag_reg * v

class S():
    def __init__(self, f, fav, wf, Ns, diag_reg=1e-4):
        # f: the function used to define the variational derivatives on
        self.f = f
        self.fav = fav
        self.wf = wf
        self.Ns = Ns
        self.diag_reg = diag_reg

    def __matmul__(self, v):
        _, jvp = torch.func.jvp(self.f, (self.wf.get_params(),), (v,))
        _, vjp = torch.func.vjp(self.f, self.wf.get_params())

        res = vjp(jvp)[0]/self.Ns

        _, jvp = torch.func.jvp(self.fav, (self.wf.get_params(),), (v,))
        _, vjp = torch.func.vjp(self.fav, self.wf.get_params())
        return res - vjp(jvp)[0] - self.diag_reg * v