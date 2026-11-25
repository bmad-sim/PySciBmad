import scibmad as sb
import numpy as np
import pytest
import torch

from scibmad._torch_bridge import _resolve_backend
from scibmad._runtime import jl


@pytest.mark.parametrize("backend_name", ["forwarddiff", "gtpsa", "reversediff"])
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

def test_multiparticle_vjp_against_finite_difference():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)
    d = sb.Drift(L=1.2)
    qd = sb.Quadrupole(L=0.5, Kn1=-0.36)

    line = sb.Beamline(
        [qf, d, qd, d],
        E_ref=18e9,
        species_ref=sb.Species("electron"),
    )

    coords = torch.tensor(
        [
            [1e-3, 2e-4, 1e-3, 0, 0, 0],
            [2e-3, 0, -1e-3, 1e-4, 0, 0],
            [-1e-3, 1e-4, 2e-3, 0, 0, 0],
        ],
        dtype=torch.float64,
    )

    def loss_fn(k):
        output = sb.torch_track(
            line,
            v0=coords,
            parameters=[(qf, "Kn1", k)],
            # test backends here
            backend="forwarddiff",
        )
        return output[:, 0].square().sum()

    k = torch.tensor(
        0.36,
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = loss_fn(k)
    loss.backward()

    ad_grad = k.grad.item()

    # Central finite difference
    h = 1e-5

    with torch.no_grad():
        k_plus = torch.tensor(0.36 + h, dtype=torch.float64)
        k_minus = torch.tensor(0.36 - h, dtype=torch.float64)

        fd_grad = (
            loss_fn(k_plus).item()
            - loss_fn(k_minus).item()
        ) / (2 * h)

    np.testing.assert_allclose(
        ad_grad,
        fd_grad,
        rtol=1e-4,
        atol=1e-9,
    )

