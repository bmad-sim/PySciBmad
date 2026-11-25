import numpy as np
import torch
import scibmad as sb
import pytest

def test_reversediff_rejects_trainable_parameters():
    qf = sb.Quadrupole(L=0.5, Kn1=0.36)

    line = sb.Beamline(
        [qf],
        E_ref=18e9,
        species_ref=sb.Species("electron"),
    )

    coords = torch.zeros((1, 6), dtype=torch.float64)

    k1 = torch.tensor(
        0.36,
        dtype=torch.float64,
        requires_grad=True,
    )

    with pytest.raises(
        ValueError,
        match="ReverseDiff differentiation of beamline parameters",
    ):
        sb.torch_track(
            line,
            v0=coords,
            parameters=[(qf, "Kn1", k1)],
            backend="reversediff",
        )

