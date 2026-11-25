
import scibmad as sb
import torch

torch.set_default_dtype(torch.float64)


def test_sextupole_gradient():
    sext = sb.Sextupole(
        L=0.3,
        Kn2=1.2,
        name="SF_TEST",
    )
    drift = sb.Drift(L=0.5)

    bl = sb.Beamline(
        [sext, drift],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    coords = torch.tensor([
        2e-3, 0.0,
        1e-3, 0.0,
        0.0, 0.0,
    ])

    k2 = torch.tensor(
        1.2,
        requires_grad=True,
    )

    def track(k):
        return sb.torch_track(
            bl,
            v0=coords,
            parameters=[(sext, "Kn2", k)],
        )[0]

    result = track(k2)
    result.backward()

    h = 1e-5

    with torch.no_grad():
        fd = (track(k2 + h) - track(k2 - h)) / (2 * h)

    print("AD gradient:", k2.grad.item())
    print("FD gradient:", fd.item())

    torch.testing.assert_close(
        k2.grad,
        fd,
        rtol=1e-5,
        atol=1e-10,
    )

    assert float(bl.line[0].Kn2) == 1.2
