
from scibmad._runtime import jl
from scibmad._torch_bridge import _JuliaTorchCallable
import torch

jl.seval("scibmad_test_square(x) = x^2")

x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

f = _JuliaTorchCallable(jl.scibmad_test_square)

y = f(x)
y.backward()

print("Result:", y.item())
print("Gradient:", x.grad.item())

