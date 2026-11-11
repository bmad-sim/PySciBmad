
"""PyTorch-compatible native SciBmad tracking."""

from __future__ import annotations

import os

from ._runtime import glue, jl
from ._wrappers import unwrap
from ._torch_bridge import _JuliaTorchCallable

# Load the tracking helper into PySciBmad's glue module.
_tracking_path = os.path.join(
    os.path.dirname(__file__),
    "_torch_tracking.jl",
)

jl.Base.include(glue, _tracking_path)


def torch_track(
    bl,
    *,
    elements,
    v0,
    quadrupole_index,
    kn1,
):
    """Track a particle with a differentiable quadrupole strength.

    Args:
        bl: Existing PySciBmad beamline.
        elements: Original elements in beamline order.
        v0: Initial six-dimensional particle coordinates.
        quadrupole_index: Zero-based index of the quadrupole.
        kn1: Differentiable quadrupole strength.

    Returns:
        A PyTorch tensor of final particle coordinates.
    """
    import torch

    if not isinstance(v0, torch.Tensor):
        raise TypeError("v0 must be a torch.Tensor")

    if v0.ndim != 1 or v0.numel() != 6:
        raise ValueError("v0 must have shape (6,)")

    if not isinstance(kn1, torch.Tensor) or kn1.numel() != 1:
        raise TypeError("kn1 must be a scalar torch.Tensor")

    if not 0 <= quadrupole_index < len(elements):
        raise IndexError("quadrupole_index is out of range")

    # Convert the element sequence into a Julia vector.
    jl_elements = glue.to_any_vector(
        [unwrap(ele) for ele in elements]
    )

    track_fn = _JuliaTorchCallable(
        glue.torch_track_quadrupole
    )

    return track_fn(
        jl_elements,
        quadrupole_index + 1,
        v0,
        kn1,
        bl.species_ref,
        bl.p_over_q_ref,
    )
