import torch
from icecream import ic

x = torch.randn(5, requires_grad=True, dtype=torch.complex128)  # input tensor

def e(x):
    return torch.norm(x)

x.grad = None
e(x).real.backward()
ere = x.grad

# x.grad = None
# e(x).imag.backward()
# eim = x.grad

ic(ere - x / torch.norm(x))

def e(x):
    return x @ x

x.grad = None
e(x).real.backward()
ere = x.grad

x.grad = None
e(x).imag.backward()
eim = x.grad

ic(ere + eim*1j)