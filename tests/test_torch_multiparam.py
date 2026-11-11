
import scibmad as sb
import torch

torch.set_default_dtype(torch.float64)


def test_two_quadrupole_gradients():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)
    d = sb.Drift(L=1.0)
    qd = sb.Quadrupole(L=0.5, Kn1=-0.36)

    # Use separate drift objects to keep the test simple.
    d2 = sb.Drift(L=1.0)

    elements = [qf, d, qd, d2]

    bl = sb.Beamline(
        elements,
        species_ref=sb.Species("electron"),
        E_ref=18e9,
    )

    coords = torch.tensor([
        1e-3, 0.0,
        1e-3, 0.0,
        0.0, 0.0,
    ])

    k1 = torch.tensor(0.36, requires_grad=True)
    k2 = torch.tensor(-0.36, requires_grad=True)

    def track(a, b):
        return sb.torch_track(
            bl,
            v0=coords,
            parameters=[
                (qf, "Kn1", a),
                (qd, "Kn1", b),
            ],
        )

    result = track(k1, k2)

    # Differentiate final horizontal position.
    loss = result[0]
    loss.backward()

    print("Final coordinates:", result)
    print("Gradient QF:", k1.grad.item())
    print("Gradient QD:", k2.grad.item())

    assert result.shape == (6,)
    assert k1.grad is not None
    assert k2.grad is not None
    assert torch.isfinite(k1.grad)
    assert torch.isfinite(k2.grad)

    # Compare against central finite differences.
    h = 1e-5

    with torch.no_grad():
        fd1 = (
            track(k1 + h, k2)[0]
            - track(k1 - h, k2)[0]
        ) / (2 * h)

        fd2 = (
            track(k1, k2 + h)[0]
            - track(k1, k2 - h)[0]
        ) / (2 * h)

    print("FD gradient QF:", fd1.item())
    print("FD gradient QD:", fd2.item())

    torch.testing.assert_close(
        k1.grad, fd1, rtol=1e-5, atol=1e-9
    )
    torch.testing.assert_close(
        k2.grad, fd2, rtol=1e-5, atol=1e-9
    )
