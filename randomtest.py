
import scibmad as sb
import torch

torch.set_default_dtype(torch.float64)

# -----------------------------
# 1. Construct the FODO cell
# -----------------------------

qf = sb.Quadrupole(L=0.5, Kn1=0.36)
d1 = sb.Drift(L=1.0)
qd = sb.Quadrupole(L=0.5, Kn1=-0.36)
d2 = sb.Drift(L=1.0)

bl = sb.Beamline(
    [qf, d1, qd, d2],
    species_ref=sb.Species("electron"),
    E_ref=18e9,
)

coords0 = torch.tensor([
    1e-3, 0.0,
    1e-3, 0.0,
    0.0, 0.0,
])

# -----------------------------
# 2. Define the target
# -----------------------------

true_k1 = 0.36
true_k2 = -0.36

# Two different initial particles give us
# more information for fitting both strengths.
initial_particles = [
    coords0,
    torch.tensor([
        -1.5e-3, 2e-4,
        0.8e-3, -1e-4,
        0.0, 0.0,
    ]),
]

def track_particle(coords, a, b):
    return sb.torch_track(
        bl,
        v0=coords,
        parameters=[
            (qf, "Kn1", a),
            (qd, "Kn1", b),
        ],
    )

with torch.no_grad():
    targets = [
        track_particle(x, true_k1 * torch.ones(()),
                       true_k2 * torch.ones(()))
        for x in initial_particles
    ]

# -----------------------------
# 3. Introduce strength errors
# -----------------------------

k1 = torch.tensor(0.45, requires_grad=True)
k2 = torch.tensor(-0.25, requires_grad=True)

optimizer = torch.optim.Adam([k1, k2], lr=0.01)

# -----------------------------
# 4. Optimize
# -----------------------------

for iteration in range(500):
    optimizer.zero_grad()

    predictions = [
        track_particle(x, k1, k2)
        for x in initial_particles
    ]

    # Fit the first four phase-space coordinates.
    loss = sum(
        ((pred[:4] - target[:4]) ** 2).sum()
        for pred, target in zip(predictions, targets)
    )

    loss.backward()
    optimizer.step()

    if iteration % 50 == 0:
        print(
            f"Iteration {iteration:3d} | "
            f"Loss: {loss.item():.6e} | "
            f"QF: {k1.item():.6f} | "
            f"QD: {k2.item():.6f}"
        )

# -----------------------------
# 5. Results
# -----------------------------

print("\nTrue strengths:", true_k1, true_k2)
print("Optimized strengths:", k1.item(), k2.item())

print("QF error:", abs(k1.item() - true_k1))
print("QD error:", abs(k2.item() - true_k2))


"""
import scibmad as sb

sb.jl.seval(""""""
using ForwardDiff

function test_element_copy_tracking()
    q = SciBmad.Quadrupole(
        L=0.5,
        Kn1=0.36,
        name="QF_TEST",
    )

    d = SciBmad.Drift(L=1.0)
    species = SciBmad.Species("electron")

    function track_with_strength(k)
        q_new = deepcopy(q)
        q_new.Kn1 = k

        bl = SciBmad.Beamline(
            [q_new, d];
            species_ref=species,
            p_over_q_ref=1.0,
        )

        T = typeof(k)

        coords = zeros(T, 1, 6)
        coords[1, 1] = T(1e-3)
        coords[1, 3] = T(1e-3)

        bunch = SciBmad.Bunch(
            coords;
            species=bl.species_ref,
            p_over_q_ref=bl.p_over_q_ref,
            t_ref=zero(T),
        )

        SciBmad.track!(bunch, bl)

        return bunch.coords.v[1, 1]
    end

    gradient = ForwardDiff.derivative(
        track_with_strength,
        0.36,
    )

    return gradient, q.Kn1, q.name
end
"""""")

gradient, original_k1, original_name = (
    sb.jl.test_element_copy_tracking()
)

print("Tracking gradient:", gradient)
print("Original Kn1:", original_k1)
print("Original name:", original_name)

"""

"""
import scibmad as sb

sb.jl.seval(""""""
using ForwardDiff

function test_beamline_child_copy()
    qf = SciBmad.Quadrupole(
        L=0.5,
        Kn1=0.36,
        name="QF_TEST",
    )

    d = SciBmad.Drift(L=1.0)
    species = SciBmad.Species("electron")

    original_bl = SciBmad.Beamline(
        [qf, d];
        species_ref=species,
        p_over_q_ref=1.0,
    )

    function track_with_strength(k)
        # Extract elements from the existing beamline.
        elements = original_bl.line

        # Copy the beamline elements.
        new_elements = deepcopy(collect(Any, elements))

        # Modify only the desired parameter.
        new_elements[1].Kn1 = k

        # Construct another beamline.
        bl = SciBmad.Beamline(
            new_elements;
            species_ref=species,
            p_over_q_ref=1.0,
        )

        T = typeof(k)

        coords = zeros(T, 1, 6)
        coords[1, 1] = T(1e-3)
        coords[1, 3] = T(1e-3)

        bunch = SciBmad.Bunch(
            coords;
            species=bl.species_ref,
            p_over_q_ref=bl.p_over_q_ref,
            t_ref=zero(T),
        )

        SciBmad.track!(bunch, bl)

        return bunch.coords.v[1, 1]
    end

    gradient = ForwardDiff.derivative(
        track_with_strength,
        0.36,
    )

    return gradient, original_bl.line[1].Kn1
end
"""""")

gradient, original_strength = (
    sb.jl.test_beamline_child_copy()
)

print("Gradient:", gradient)
print("Original strength:", original_strength)

"""


import scibmad as sb

sb.jl.seval("""
using ForwardDiff

function test_sextupole_copy_ad()
    sext = SciBmad.Sextupole(
        L=0.3,
        Kn2=1.2,
        name="SF_TEST",
    )

    drift = SciBmad.Drift(L=0.5)
    species = SciBmad.Species("electron")

    original_bl = SciBmad.Beamline(
        [sext, drift];
        species_ref=species,
        p_over_q_ref=1.0,
    )

    function track_with_kn2(k)
        elements = deepcopy(collect(Any, original_bl.line))
        elements[1].Kn2 = k

        bl = SciBmad.Beamline(
            elements;
            species_ref=species,
            p_over_q_ref=original_bl.p_over_q_ref,
        )

        T = typeof(k)

        coords = zeros(T, 1, 6)
        coords[1, 1] = T(2e-3)
        coords[1, 3] = T(1e-3)

        bunch = SciBmad.Bunch(
            coords;
            species=bl.species_ref,
            p_over_q_ref=bl.p_over_q_ref,
            t_ref=zero(T),
        )

        SciBmad.track!(bunch, bl)

        return bunch.coords.v[1, 1]
    end

    gradient = ForwardDiff.derivative(
        track_with_kn2,
        1.2,
    )

    return gradient, original_bl.line[1].Kn2
end
""")

gradient, original_kn2 = sb.jl.test_sextupole_copy_ad()

print("Sextupole gradient:", gradient)
print("Original Kn2:", original_kn2)

assert abs(float(gradient)) > 1e-15
assert float(original_kn2) == 1.2