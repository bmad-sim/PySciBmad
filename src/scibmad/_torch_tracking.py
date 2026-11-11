
"""PyTorch-compatible native SciBmad tracking."""

from __future__ import annotations

import os

from ._runtime import glue, jl
from ._wrappers import unwrap
from ._torch_bridge import _JuliaTorchCallable

import torch


_tracking_path = os.path.join(
    os.path.dirname(__file__),
    "_torch_tracking.jl",
)

jl.Base.include(glue, _tracking_path)


def torch_track(
    bl,
    *,
    v0,
    parameters=None,
    quadrupole_index=None,
    kn1=None,
):
    """Differentiable tracking through a SciBmad beamline.

    parameters:
        Sequence of (element, "Kn1", tensor) tuples.

    Legacy interface:
        quadrupole_index and kn1 are also supported.
    """

    if not isinstance(v0, torch.Tensor):
        raise TypeError("v0 must be a torch.Tensor")

    # Accept a single particle (6,) or multiple particles (N, 6)
    if v0.ndim == 1:
        if v0.shape != (6,):
            raise ValueError(
                "Single-particle input must have shape (6,)"
            )

    elif v0.ndim == 2:
        if v0.shape[1] != 6 or v0.shape[0] == 0:
            raise ValueError(
                "Multi-particle input must have shape (N, 6), N >= 1"
            )

    else:
        raise ValueError(
            "v0 must have shape (6,) or (N, 6)"
        )


    if not v0.is_floating_point():
        raise TypeError("v0 must have a floating-point dtype")

    elements = list(bl.line)

    # Preserve the previous single-quadrupole API.
    if parameters is None:
        if quadrupole_index is None or kn1 is None:
            raise ValueError(
                "Provide parameters or quadrupole_index and kn1"
            )

        if not 0 <= quadrupole_index < len(elements):
            raise IndexError("quadrupole_index is out of range")

        parameters = [
            (elements[quadrupole_index], "Kn1", kn1)
        ]
    elif quadrupole_index is not None or kn1 is not None:
        raise ValueError(
            "Do not combine parameters with legacy arguments"
        )

    if not parameters:
        raise ValueError("parameters must not be empty")

    indices = []
    names = []
    values = []

    for element, attribute, value in parameters:
        if not isinstance(attribute, str) or not attribute:
            raise TypeError("Parameter name must be a nonempty string")

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "Parameter values must be torch.Tensor objects"
            )

        if value.numel() != 1:
            raise ValueError(
                "Each parameter must be a scalar tensor"
            )

        if value.dtype != v0.dtype or value.device != v0.device:
            raise ValueError(
                "Parameter dtype and device must match v0"
            )

        # Identify the element by Python object identity.
        # Find child elements belonging to the original parent.
        getindex = jl.seval("Base.getindex")
        children = getindex(bl, unwrap(element))

        matches = [
            int(child.beamline_index) - 1
            for child in children
        ]

        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one child instance for this parameter; "
                f"found {len(matches)}"
            )

        if len(matches) != 1:
            raise ValueError(
                "Each parameter must identify exactly one "
                "element in the supplied elements list"
            )

        index = matches[0]

        if (index, attribute) in zip(indices, names):
            raise ValueError(
                f"Parameter {attribute!r} was specified more than once "
                f"for element at index {index}"
            )
        
        indices.append(index)
        names.append(attribute)
        values.append(value.reshape(()))

    # Stack scalar tensors into one differentiable vector.
    strengths = torch.stack(values)

    # Convert Python elements and indices to Julia arrays.
    jl_elements = glue.to_any_vector(
        [unwrap(ele) for ele in elements]
    )

    jl_indices = glue.to_any_vector(
        [i + 1 for i in indices]
    )

    track_fn = _JuliaTorchCallable(
        glue.torch_track_parameters
    )

    jl_names = glue.to_any_vector(names)

    # Flatten coordinates without detaching them to preserve the PyTorch computation graph.
    flat_coords = v0.reshape(-1)

    flat_result = track_fn(
        jl_elements,
        jl_indices,
        jl_names,
        flat_coords,
        strengths,
        bl.species_ref,
        bl.p_over_q_ref,
    )

    # Restore the original particle dimensions
    return flat_result.reshape(v0.shape)
