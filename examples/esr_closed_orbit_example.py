"""
Closed-Orbit Optimization of the 18 GeV ESR using PySciBmad
"""

from scibmad._runtime import jl
import scibmad as sb
import torch
import numpy as np

# Requires the publicly available ESR lattice file:
# esr-v6.3.1.jl
# Place it in your working directory before running this example.

# Load ESR Lattice

jl.include("esr-v6.3.1.jl")

ring = jl.ring

print("18 GeV ESR loaded")
print("Number of elements:", len(ring.line))


# Define C.O. Objective

def orbit_loss(coords):
    """
    Minimize the one-turn transverse phase-space residual
    """

    # Longitudinal coordinates remain fixed
    v0 = torch.cat([
        coords,
        torch.zeros(2, dtype=torch.float64),
    ])

    final = sb.torch_track(
        ring,
        v0=v0,
        backend="reversediff",
    )

    residual = final[:4] - coords

    return torch.sum(residual**2)

# Optimize initial transverse coordinates

coords = torch.nn.Parameter(
    torch.tensor(
        [1e-3, 0.0, 1e-3, 0.0],
        dtype=torch.float64,
    )
)

optimizer = torch.optim.Adam([coords], lr=1e-5)

n_steps = 100

print("\nStarting closed-orbit optimization")

for step in range(n_steps):

    optimizer.zero_grad()

    loss = orbit_loss(coords)

    loss.backward()

    optimizer.step()

    if (step + 1) % 10 == 0 or step == 0:
        print(
            f"Step {step+1:3d} | "
            f"Loss = {loss.item():.10e}"
        )

# Evaluate optimized orbit

with torch.no_grad():

    optimized_coords = coords.detach().clone()

    optimized_v0 = torch.cat([
        optimized_coords,
        torch.zeros(2, dtype=torch.float64),
    ])

    final = sb.torch_track(
        ring,
        v0=optimized_v0,
        backend="reversediff",
    )

    residual = final[:4] - optimized_coords
    final_loss = torch.sum(residual**2)

print("\nOptimization results")

print("Optimized initial coordinates:", optimized_coords)

print("Final loss:", final_loss.item())

print("One-turn transverse residual:", residual)

print(
    "Maximum absolute transverse residual:",
    torch.max(torch.abs(residual)).item(),
)

# Compare against native SciBmad

native_orbit = jl.SciBmad.find_closed_orbit(ring)

native_coords = np.array(native_orbit.v0)[0, :4]

print("\nNative SciBmad closed orbit:", native_coords)

# Check against native solution 

with torch.no_grad():

    native_v0 = torch.tensor(
        np.array(native_orbit.v0).reshape(-1),
        dtype=torch.float64,
    )

    native_final = sb.torch_track(
        ring,
        v0=native_v0,
        backend="reversediff",
    )

    native_residual = native_final[:4] - native_v0[:4]

print(
    "PySciBmad residual at native orbit:",
    native_residual,
)