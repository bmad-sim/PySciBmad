
import scibmad as sb
import torch

torch.set_default_dtype(torch.float64)

# Construct a native beamline
qf = sb.Quadrupole(L=0.5, Kn1=0.36)
d = sb.Drift(L=1.0)

elements = [qf, d]

bl = sb.Beamline(
    elements,
    species_ref=sb.Species("electron"),
    p_over_q_ref=1.0,
)

# Initial coordinates
coords = torch.tensor([
    1e-3, 0.0,
    1e-3, 0.0,
    0.0, 0.0,
])

# Differentiable quadrupole strength
k1 = torch.tensor(
    0.36,
    requires_grad=True,
)

# Tracking through native SciBmad
result = sb.torch_track(
    bl,
    v0=coords,
    quadrupole_index=0,
    kn1=k1,
)

# Differentiate horizontal position
loss = result[0]
loss.backward()

print("Final coordinates:", result)
print("Gradient:", k1.grad.item())

assert result.shape == (6,)
assert k1.grad is not None
assert torch.isfinite(k1.grad)

h = 1e-5

def final_x(strength):
    with torch.no_grad():
        return sb.torch_track(
            bl,
            v0=coords,
            quadrupole_index=0,
            kn1=torch.tensor(strength),
        )[0].item()

grad_fd = (final_x(0.36 + h) - final_x(0.36 - h)) / (2 * h)

assert abs(k1.grad.item() - grad_fd) < 1e-8