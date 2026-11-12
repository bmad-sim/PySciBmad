using SciBmad
import GTPSA

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

   
    # Determine the number of particles.
    length(coords) % 6 == 0 ||
        throw(ArgumentError(
            "Particle coordinate count must be divisible by 6"
        ))

    n_particles = length(coords) ÷ 6

    n_particles > 0 ||
        throw(ArgumentError(
            "At least one particle is required"
        ))

    # Preserve the ForwardDiff dual-number type.
    T = promote_type(eltype(coords), eltype(strengths))

    # Convert the flattened coordinates into
    # an N x 6 Julia matrix.
    v = permutedims(
        reshape(T.(coords), 6, n_particles)
    )

    # Track all particles as one native SciBmad Bunch.
    bunch = SciBmad.Bunch(
        v;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T),
    )

    SciBmad.track!(bunch, bl)

    # Flatten in particle-major order for Python.
    return vec(permutedims(bunch.coords.v))
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

    length(coords) % 6 == 0 ||
        throw(ArgumentError("Coordinate count must be divisible by 6"))

    n_particles = length(coords) ÷ 6

    n_particles > 0 ||
        throw(ArgumentError("At least one particle is required"))

    # Convert flattened coordinates into an N x 6 matrix.
    v = permutedims(reshape(T.(coords), 6, n_particles))

    bunch = SciBmad.Bunch(
        v;
        species=bl.species_ref,
        p_over_q_ref=bl.p_over_q_ref,
        t_ref=zero(T),
    )

    SciBmad.track!(bunch, bl)

    return vec(permutedims(bunch.coords.v))
end


function torch_track_gtpsa_jacobian(
    elements,
    element_indices,
    parameter_names,
    coords,
    strengths,
    species_ref,
    p_over_q_ref,
)
    n = length(strengths)

    n > 0 ||
        throw(ArgumentError("At least one parameter is required"))

    # First-order Taylor expansion in the selected strengths.
    d = GTPSA.Descriptor(n, 1)
    delta = GTPSA.vars(d)

    tps_strengths = [
        GTPSA.TPS(Float64(strengths[j]); use=d) + delta[j]
        for j in 1:n
    ]

    # Reuse the existing native SciBmad tracking function.
    output = torch_track_parameters(
        elements,
        element_indices,
        parameter_names,
        coords,
        tps_strengths,
        species_ref,
        p_over_q_ref,
    )

    # Extract the numerical output and selected coefficients.
    values = Vector{Float64}(undef, length(output))
    jac = Matrix{Float64}(undef, length(output), n)

    for i in eachindex(output)
        values[i] = output[i][0]

        for j in 1:n
            jac[i, j] = output[i][j]
        end
    end

    return values, jac
end


