
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
