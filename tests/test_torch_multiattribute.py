import scibmad as sb
import torch


def test_multiple_parameters_same_element():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36, Ks1=0.02)
    d = sb.Drift(L=1.0)

    bl = sb.Beamline(
        [qf, d],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    coords = torch.tensor(
        [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )

    kn1 = torch.tensor(0.36, dtype=torch.float64, requires_grad=True)
    ks1 = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)

    result = sb.torch_track(
        bl,
        v0=coords,
        parameters=[
            (qf, "Kn1", kn1),
            (qf, "Ks1", ks1),
        ],
    )

    loss = result[:4].sum()
    loss.backward()

    assert kn1.grad is not None
    assert ks1.grad is not None
    assert torch.isfinite(kn1.grad)
    assert torch.isfinite(ks1.grad)