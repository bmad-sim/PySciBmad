import pytest
import scibmad as sb


def test_tracker_backend_configuration():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)

    bl = sb.Beamline(
        [qf],
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    tracker = sb.TorchTracker(
        bl,
        backend="forwarddiff",
    )

    assert tracker.beamline is bl
    assert tracker.backend == "forwarddiff"

    with pytest.raises(ValueError, match="Unsupported AD backend"):
        sb.TorchTracker(bl, backend="invalid")
