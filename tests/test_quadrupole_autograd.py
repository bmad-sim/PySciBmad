
import scibmad as sb
from scibmad._torch_bridge import _JuliaTorchCallable
import torch

# Define the tracking operation directly in Julia.
# The trainable parameter k1 is passed as a function argument.

sb.jl.seval("""
using SciBmad

function test_quadrupole_track(k1)
    T = typeof(k1)

    qf = Quadrupole(L=T(0.5), Kn1=k1)
    d = Drift(L=T(1.0))

    species = Species("electron")
    p_ref = T(1.0)

    bl = Beamline(
        [qf, d];
        species_ref=species,
        p_over_q_ref=p_ref
    )

    coords = zeros(T, 1, 6)
    coords[1, 1] = T(1e-3)
    coords[1, 3] = T(1e-3)

    bunch = Bunch(
        coords;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T)
    )

    track!(bunch, bl)

    return bunch.coords.v[1, 1]
end
""")

# Wrap the Julia tracking function with PyTorch autograd
track = _JuliaTorchCallable(sb.jl.test_quadrupole_track)

# Trainable quadrupole strength
k1 = torch.tensor(
    0.36,
    dtype=torch.float64,
    requires_grad=True
)

# Forward pass
x_final = track(k1)

# Backward pass
x_final.backward()

print("Final x:", x_final.item())
print("dx/dKn1:", k1.grad.item())

assert torch.isfinite(x_final)
assert k1.grad is not None
assert torch.isfinite(k1.grad)


# Compare autodiff against central finite differences
h = 1e-5
k = k1.detach().item()

x_plus = float(sb.jl.test_quadrupole_track(k + h))
x_minus = float(sb.jl.test_quadrupole_track(k - h))

grad_fd = (x_plus - x_minus) / (2 * h)
grad_ad = k1.grad.item()

print("AD gradient:", grad_ad)
print("FD gradient:", grad_fd)
print("Absolute difference:", abs(grad_ad - grad_fd))

assert abs(grad_ad - grad_fd) < 1e-8

