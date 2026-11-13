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

def test_gtpsa_multiparticle_multiparameter():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)
    qd = sb.Quadrupole(L=0.5, Kn1=-0.36)

    bl = sb.Beamline(
        [qf, sb.Drift(L=1.0), qd],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    coords = torch.tensor(
        [
            [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
            [-2e-3, 1e-4, 0.5e-3, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    def gradients(backend):
        kf = torch.tensor(0.36, dtype=torch.float64, requires_grad=True)
        kd = torch.tensor(-0.36, dtype=torch.float64, requires_grad=True)

        tracker = sb.TorchTracker(bl, backend=backend)
        result = tracker(
            v0=coords,
            parameters=[
                (qf, "Kn1", kf),
                (qd, "Kn1", kd),
            ],
        )

        loss = result[:, 0].square().sum()
        loss.backward()

        return result.detach(), kf.grad.detach(), kd.grad.detach()

    result_fd, gf_fd, gd_fd = gradients("forwarddiff")
    result_gt, gf_gt, gd_gt = gradients("gtpsa")

    torch.testing.assert_close(result_gt, result_fd, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(gf_gt, gf_fd, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(gd_gt, gd_fd, rtol=1e-5, atol=1e-9)
