
import scibmad as sb
from scibmad._torch_bridge import _JuliaTorchCallable
import torch

torch.set_default_dtype(torch.float64)

# Construct beamline through PySciBmad's existing API
qf = sb.Quadrupole(L=0.5, Kn1=0.36)
d = sb.Drift(L=1.0)

bl = sb.Beamline(
    [qf, d],
    species_ref=sb.Species("electron"),
    p_over_q_ref=1.0,
)

# Julia tracking helper
sb.jl.seval("""
function scibmad_track_coords(bl, coords)
    T = eltype(coords)

    v = Matrix{T}(undef, 1, 6)
    v[1, :] .= coords

    bunch = SciBmad.Bunch(
        v;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T)
    )

    SciBmad.track!(bunch, bl)

    return vec(bunch.coords.v[1, :])
end
""")

track = _JuliaTorchCallable(sb.jl.scibmad_track_coords)

coords = torch.tensor(
    [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
    requires_grad=True
)

# Forward tracking through native beamline
result = track(bl, coords)

# Differentiate final horizontal position
loss = result[0]
loss.backward()

print("Final coordinates:", result)
print("Gradient w.r.t. initial coordinates:", coords.grad)

assert result.shape == (6,)
assert coords.grad is not None
assert torch.all(torch.isfinite(coords.grad))
