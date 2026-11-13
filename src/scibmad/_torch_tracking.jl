using SciBmad
import GTPSA

# Track through a beamline with multiple differentiable
# quadrupole strengths.

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


