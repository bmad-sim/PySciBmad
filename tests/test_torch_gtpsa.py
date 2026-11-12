import scibmad as sb
import torch


def test_gtpsa_matches_forwarddiff():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)

    bl = sb.Beamline(
        [qf, sb.Drift(L=1.0)],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    coords = torch.tensor(
        [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )

    def evaluate(backend):
        k = torch.tensor(
            0.36,
            dtype=torch.float64,
            requires_grad=True,
        )

        tracker = sb.TorchTracker(bl, backend=backend)

        result = tracker(
            v0=coords,
            parameters=[(qf, "Kn1", k)],
        )

        loss = result[0]
        loss.backward()

        return result.detach(), k.grad.detach()

    result_fd, grad_fd = evaluate("forwarddiff")
    result_gtpsa, grad_gtpsa = evaluate("gtpsa")

    torch.testing.assert_close(
        result_gtpsa,
        result_fd,
        rtol=1e-9,
        atol=1e-12,
    )

    torch.testing.assert_close(
        grad_gtpsa,
        grad_fd,
        rtol=1e-6,
        atol=1e-10,
    )
