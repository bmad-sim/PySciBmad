
import scibmad as sb
import torch

torch.set_default_dtype(torch.float64)


def test_multiparticle_autograd():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)
    drift = sb.Drift(L=1.0)

    bl = sb.Beamline(
        [qf, drift],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    # Two distinct initial particles.
    coords = torch.tensor(
        [
            [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
            [-2e-3, 1e-4, 0.5e-3, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )

    k1 = torch.tensor(
        0.36,
        requires_grad=True,
    )

    def track(x, k):
        return sb.torch_track(
            bl,
            v0=x,
            parameters=[(qf, "Kn1", k)],
        )

    # Track both particles together.
    result = track(coords, k1)

    print("Final coordinates:")
    print(result)

    assert result.shape == (2, 6)

    # Track particles independently.
    with torch.no_grad():
        separate = torch.stack([
            track(coords[0], k1),
            track(coords[1], k1),
        ])

    torch.testing.assert_close(
        result.detach(),
        separate,
        rtol=1e-10,
        atol=1e-12,
    )

    # Backpropagate through both particles.
    loss = result[:, 0].sum()
    loss.backward()

    print("Gradient w.r.t. Kn1:", k1.grad)
    print("Gradient w.r.t. coordinates:")
    print(coords.grad)

    assert k1.grad is not None
    assert coords.grad is not None
    assert coords.grad.shape == (2, 6)

    assert torch.isfinite(k1.grad)
    assert torch.all(torch.isfinite(coords.grad))

    # Verify parameter gradient with finite differences.
    h = 1e-5

    with torch.no_grad():
        fd = (
            track(coords, k1 + h)[:, 0].sum()
            - track(coords, k1 - h)[:, 0].sum()
        ) / (2 * h)

    print("AD gradient:", k1.grad.item())
    print("FD gradient:", fd.item())

    torch.testing.assert_close(
        k1.grad,
        fd,
        rtol=1e-5,
        atol=1e-9,
    )
