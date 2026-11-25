import scibmad as sb
import numpy as np
import torch

from scibmad._runtime import jl, glue
from scibmad._wrappers import unwrap
from scibmad import _torch_tracking  # Loads Julia tracking functions


def test_gtpsa_tracking_coefficients():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)
    qd = sb.Quadrupole(L=0.5, Kn1=-0.36)

    bl = sb.Beamline(
        [qf, sb.Drift(L=1.0), qd],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    coords = np.array([
        [1e-3, 0, 1e-3, 0, 0, 0],
        [-2e-3, 1e-4, 5e-4, 0, 0, 0],
    ], dtype=np.float64)

    # Direct GTPSA coefficient extraction.
    values, jac = glue.torch_track_gtpsa_jacobian(
        glue.to_any_vector([
            unwrap(e) for e in list(bl.line)
        ]),
        jl.seval("[1, 3]"),
        jl.seval('["Kn1", "Kn1"]'),
        coords.reshape(-1),
        np.array([0.36, -0.36]),
        bl.species_ref,
        bl.p_over_q_ref,
    )

    values = np.asarray(values, dtype=np.float64)
    jac = np.asarray(jac, dtype=np.float64)

    assert jac.shape == (12, 2)

    # Compare with existing differentiable SciBmad tracking.
    kf = torch.tensor(0.36, dtype=torch.float64, requires_grad=True)
    kd = torch.tensor(-0.36, dtype=torch.float64, requires_grad=True)

    tracker = sb.TorchTracker(bl, backend="gtpsa")

    result = tracker(
        v0=torch.tensor(coords, dtype=torch.float64),
        parameters=[
            (qf, "Kn1", kf),
            (qd, "Kn1", kd),
        ],
    )

    weights = torch.arange(
        1, 13, dtype=torch.float64
    )

    loss = (result.reshape(-1) * weights).sum()
    loss.backward()

    expected = np.array([kf.grad.item(), kd.grad.item()])
    extracted = jac.T @ weights.numpy()

    np.testing.assert_allclose(
        values,
        result.detach().numpy().reshape(-1),
        rtol=1e-9,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        extracted,
        expected,
        rtol=1e-6,
        atol=1e-9,
    )
