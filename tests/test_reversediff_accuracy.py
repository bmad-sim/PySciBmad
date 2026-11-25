
import numpy as np
import pytest
import scibmad as sb

from scibmad._runtime import jl, glue
from scibmad._torch_bridge import _resolve_backend
from scibmad._wrappers import unwrap
from scibmad import _torch_tracking


@pytest.mark.parametrize("n_particles", [1, 10, 50, 100])
def test_reversediff_coordinate_gradients(n_particles):

    rng = np.random.default_rng(42)

    # Build beamline
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)
    d = sb.Drift(L=1.2)
    qd = sb.Quadrupole(L=0.5, Kn1=-0.36)

    line = sb.Beamline(
        [qf, d, qd, d],
        E_ref=18e9,
        species_ref=sb.Species("electron"),
    )

    elements = list(line.line)
    getindex = jl.seval("Base.getindex")
    children = getindex(line, unwrap(qf))
    qf_index = int(list(children)[0].beamline_index) - 1

    args_fixed = (
        glue.to_any_vector([unwrap(e) for e in elements]),
        glue.to_any_vector([qf_index + 1]),
        glue.to_any_vector(["Kn1"]),
    )

    coords = np.zeros((n_particles, 6), dtype=np.float64)
    coords[:, 0] = rng.normal(0, 1e-3, n_particles)
    coords[:, 1] = rng.normal(0, 1e-4, n_particles)
    coords[:, 2] = rng.normal(0, 1e-3, n_particles)
    coords[:, 3] = rng.normal(0, 1e-4, n_particles)

    x = coords.reshape(-1)
    strengths = np.array([0.36], dtype=np.float64)

    args = (
        *args_fixed,
        x,
        strengths,
        line.species_ref,
        line.p_over_q_ref,
    )

    # Deterministic loss: weighted sum of final coordinates
    cotangent = rng.normal(size=x.size)

    f = glue.torch_track_parameters

    def calculate_vjp(backend_name):
        return np.asarray(
            jl.scibmad_vjp_arg(
                f,
                _resolve_backend(backend_name),
                args,
                4,
                x,
                cotangent,
            ),
            dtype=np.float64,
        )

    forward_grad = calculate_vjp("forwarddiff")
    reverse_grad = calculate_vjp("reversediff")

    np.testing.assert_allclose(
        reverse_grad,
        forward_grad,
        rtol=1e-6,
        atol=1e-9,
    )
