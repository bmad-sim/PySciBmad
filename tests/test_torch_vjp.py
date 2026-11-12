import scibmad as sb
import numpy as np
import pytest

from scibmad._torch_bridge import _resolve_backend
from scibmad._runtime import jl


@pytest.mark.parametrize("backend_name", ["forwarddiff", "gtpsa"])
def test_vjp_matches_jacobian(backend_name):
    # Julia function: R^2 -> R^2
    f = jl.seval("""
        x -> [
            x[1]^2 + 3*x[2],
            sin(x[1]*x[2])
        ]
    """)

    x = np.array([0.3, -0.2], dtype=np.float64)

    # Incoming gradient from a hypothetical loss.
    cotangent = np.array([1.5, -0.7], dtype=np.float64)

    backend = _resolve_backend(backend_name)

    # Existing full-Jacobian implementation.
    jac = np.asarray(
        jl.scibmad_jacobian_arg(
            f, backend, (x,), 1, x
        ),
        dtype=np.float64,
    )

    expected = jac.T @ cotangent

    # New direct VJP implementation.
    actual = np.asarray(
        jl.scibmad_vjp_arg(
            f, backend, (x,), 1, x, cotangent
        ),
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-8,
        atol=1e-10,
    )
