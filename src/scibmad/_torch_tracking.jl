
using SciBmad

function torch_track_quadrupole(
    elements,
    element_index,
    coords,
    kn1,
    species_ref,
    p_over_q_ref,
)
    # Reconstruct the selected quadrupole with the
    # differentiable strength.
    new_elements = collect(Any, elements)

    original = new_elements[element_index]

    new_elements[element_index] = SciBmad.Quadrupole(
        L=original.L,
        Kn1=kn1,
    )

    # Construct a fresh beamline.
    bl = SciBmad.Beamline(
        new_elements;
        species_ref=species_ref,
        p_over_q_ref=p_over_q_ref,
    )

    # Preserve the ForwardDiff dual-number type.
    T = promote_type(eltype(coords), typeof(kn1))

    v = Matrix{T}(undef, 1, 6)
    v[1, :] .= coords

    bunch = SciBmad.Bunch(
        v;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T),
    )

    SciBmad.track!(bunch, bl)

    return vec(bunch.coords.v[1, :])
end


# Track through a beamline with multiple differentiable
# quadrupole strengths.

"""
function torch_track_parameters(
    elements,
    element_indices,
    coords,
    strengths,
    species_ref,
    p_over_q_ref,
)
    length(element_indices) == length(strengths) ||
        throw(ArgumentError(
            "Number of indices must match number of strengths"
        ))

    # Deep-copy all beamline child elements.
    # The original beamline remains unchanged.
    new_elements = deepcopy(collect(Any, elements))

    # Update the selected parameters.
    for j in eachindex(element_indices)
        idx = Int(element_indices[j])

        new_elements[idx].Kn1 = strengths[j]
    end

    # Construct a new native SciBmad beamline.
    bl = SciBmad.Beamline(
        new_elements;
        species_ref=species_ref,
        p_over_q_ref=p_over_q_ref,
    )

    # Preserve ForwardDiff's dual-number types.
    T = promote_type(eltype(coords), eltype(strengths))

    v = Matrix{T}(undef, 1, 6)
    v[1, :] .= coords

    bunch = SciBmad.Bunch(
        v;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T),
    )

    SciBmad.track!(bunch, bl)

    return vec(bunch.coords.v[1, :])
end"""


function torch_track_parameters(
    elements,
    element_indices,
    parameter_names,
    coords,
    strengths,
    species_ref,
    p_over_q_ref,
)
    n = length(strengths)

    (length(element_indices) == n &&
     length(parameter_names) == n) ||
        throw(ArgumentError("Parameter lengths must match"))

    # Copy the existing beamline elements.
    new_elements = deepcopy(collect(Any, elements))

    # Apply each differentiable parameter.
    for j in 1:n
        idx = Int(element_indices[j])
        param = Symbol(parameter_names[j])

        setproperty!(new_elements[idx], param, strengths[j])
    end

    # Reconstruct the beamline.
    bl = SciBmad.Beamline(
        new_elements;
        species_ref=species_ref,
        p_over_q_ref=p_over_q_ref,
    )

    T = promote_type(eltype(coords), eltype(strengths))

    v = Matrix{T}(undef, 1, 6)
    v[1, :] .= coords

    bunch = SciBmad.Bunch(
        v;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T),
    )

    SciBmad.track!(bunch, bl)

    return vec(bunch.coords.v[1, :])
end


