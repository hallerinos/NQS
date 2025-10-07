import torch
from icecream import ic
from pyinstrument import Profiler

n = 2**29
ic(n)

# for a scalar function
x = torch.randn([n])
f = lambda x: x.sin().sum()
y = torch.randn(f(x).shape)
# ic(x, y)
(_, vjpfunc) = torch.func.vjp(f, x)
grad = vjpfunc(y)[0]
ic(torch.allclose(grad, torch.func.grad(f)(x) * y))


# for a vector field
x = torch.randn([n,])
y = torch.randn(x.shape)
# ic(x, y)
f = lambda x: x.sin()

with Profiler() as profiler:
    (_, vjpfunc) = torch.func.vjp(f, x)
    grad = vjpfunc(y)[0]
profiler.print()

with Profiler() as profiler:
    vmgrad = torch.vmap(torch.func.grad(f))
    res = vmgrad(x) * y
    # ic(grad, vmgrad(x)*y)
profiler.print()
ic(torch.allclose(grad, res))