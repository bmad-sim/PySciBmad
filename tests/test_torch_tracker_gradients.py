import torch
import scibmad as sb


def test_tracker_matches_torch_track():
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

    tracker = sb.TorchTracker(bl, backend="forwarddiff")

    def evaluate(use_tracker):
        k = torch.tensor(
            0.36, dtype=torch.float64, requires_grad=True
        )
        params = [(qf, "Kn1", k)]

        if use_tracker:
            result = tracker(v0=coords, parameters=params)
        else:
            result = sb.torch_track(
                bl, v0=coords, parameters=params,
                backend="forwarddiff",
            )

        loss = result[0]
        loss.backward()
        return result.detach(), k.grad.detach()

    result_tracker, grad_tracker = evaluate(True)
    result_direct, grad_direct = evaluate(False)

    torch.testing.assert_close(result_tracker, result_direct)
    torch.testing.assert_close(grad_tracker, grad_direct)
