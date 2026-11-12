"""Benchmark native SciBmad strength Jacobians: ForwardDiff, GTPSA, and central FD.

Run from the PySciBmad repository root:
    python3 benchmarks/benchmark_gtpsa_tracking.py --csv gtpsa_results.csv

All methods differentiate the SAME native tracking map with respect to all selected
quadrupole strengths. Timing includes Python/Julia calls and derivative computation,
but excludes package initialization and explicit warm-ups. GTPSA coefficient extraction
also returns forward coordinates. Finite differences use 2 * magnets track calls.
"""

import argparse
import csv
import statistics
import time

import numpy as np
import scibmad as sb  # Load juliacall before scibmad._torch_bridge imports torch.
import scibmad._torch_tracking  # Register Julia tracking functions in glue.
from scibmad._runtime import glue, jl
from scibmad._torch_bridge import _resolve_backend
from scibmad._wrappers import unwrap


def make_case(n_particles, n_magnets):
    magnets = [
        sb.Quadrupole(L=0.5, Kn1=0.36 if i % 2 == 0 else -0.36)
        for i in range(n_magnets)
    ]
    elements = []
    for magnet in magnets:
        elements.extend((magnet, sb.Drift(L=1.0)))

    bl = sb.Beamline(
        elements,
        species_ref=sb.Species("electron"),
        p_over_q_ref=1.0,
    )

    getindex = jl.seval("Base.getindex")
    indices = []
    for magnet in magnets:
        children = getindex(bl, unwrap(magnet))
        matches = [int(child.beamline_index) for child in children]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one child per magnet; got {len(matches)}")
        indices.append(matches[0])  # Julia 1-based indexing.

    coords = np.zeros((n_particles, 6), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0005, 0.002, n_particles)
    coords[:, 1] = np.linspace(-0.0001, 0.0001, n_particles)
    coords[:, 2] = np.linspace(0.001, -0.001, n_particles)
    strengths = np.array(
        [0.36 if i % 2 == 0 else -0.36 for i in range(n_magnets)],
        dtype=np.float64,
    )

    return (
        glue.to_any_vector([unwrap(e) for e in list(bl.line)]),
        glue.to_any_vector(indices),
        glue.to_any_vector(["Kn1"] * n_magnets),
        coords.reshape(-1),
        strengths,
        bl.species_ref,
        bl.p_over_q_ref,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--particles", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--magnets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--step", type=float, default=1e-5,
                        help="Central finite-difference step in Kn1 units")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    options = parser.parse_args()
    if options.repeats < 1 or not options.particles or not options.magnets:
        parser.error("repeats must be positive and particle/magnet lists nonempty")
    if min(*options.particles, *options.magnets) < 1 or options.step <= 0:
        parser.error("Particle/magnet counts and FD step must be positive")

    if not hasattr(glue, "torch_track_gtpsa_jacobian"):
        raise RuntimeError("Missing torch_track_gtpsa_jacobian in your local SciBmad glue")

    gtpsa_backend = _resolve_backend("gtpsa")
    forward_backend = _resolve_backend("forwarddiff")

    rows = []
    print("particles magnets   FD_ms  Forward_ms  GTPSA_DI_ms  GTPSA_coeff_ms "
          "GTPSA/FD  FD_err_max", flush=True)

    for n_particles in options.particles:
        for n_magnets in options.magnets:
            args = make_case(n_particles, n_magnets)
            strengths = args[4]
            expected_shape = (6 * n_particles, n_magnets)

            def di_jac(backend):
                return np.asarray(
                    jl.scibmad_jacobian_arg(
                        glue.torch_track_parameters, backend, args, 5, strengths
                    ), dtype=np.float64
                )

            def gtpsa_coeff():
                values, jac = glue.torch_track_gtpsa_jacobian(*args)
                return np.asarray(values, dtype=np.float64), np.asarray(jac, dtype=np.float64)

            def central_fd():
                # Reuse every non-strength input. Each column needs two native tracks.
                jac = np.empty(expected_shape, dtype=np.float64)
                for j in range(n_magnets):
                    plus = strengths.copy()
                    minus = strengths.copy()
                    plus[j] += options.step
                    minus[j] -= options.step
                    fp = np.asarray(
                        glue.torch_track_parameters(*args[:4], plus, *args[5:]),
                        dtype=np.float64,
                    ).reshape(-1)
                    fm = np.asarray(
                        glue.torch_track_parameters(*args[:4], minus, *args[5:]),
                        dtype=np.float64,
                    ).reshape(-1)
                    jac[:, j] = (fp - fm) / (2 * options.step)
                return jac

            functions = {
                "finite_diff": central_fd,
                "forwarddiff": lambda: di_jac(forward_backend),
                "gtpsa_di": lambda: di_jac(gtpsa_backend),
                "gtpsa_coeff": gtpsa_coeff,
            }
            # Warm all the methods, including JIT compilation, before timing.
            results = {name: fn() for name, fn in functions.items()}
            coeff_values, coeff_jac = results["gtpsa_coeff"]
            results["gtpsa_coeff"] = coeff_jac

            for name, jac in results.items():
                if jac.shape != expected_shape:
                    raise AssertionError(
                        f"{name}: got {jac.shape}; expected {expected_shape}"
                    )
            if coeff_values.shape != (6 * n_particles,):
                raise AssertionError(f"GTPSA output coordinate shape: {coeff_values.shape}")

            # Cross-check numerical FD against an independent AD backend.
            np.testing.assert_allclose(
                results["gtpsa_di"], results["forwarddiff"],
                rtol=1e-6, atol=1e-9,
                err_msg="GTPSA DI and ForwardDiff disagree",
            )
            np.testing.assert_allclose(
                results["gtpsa_coeff"], results["forwarddiff"],
                rtol=1e-6, atol=1e-9,
                err_msg="GTPSA coefficients and ForwardDiff disagree",
            )
            fd_error = float(np.max(np.abs(results["finite_diff"] - results["forwarddiff"])))
            np.testing.assert_allclose(
                results["finite_diff"], results["forwarddiff"],
                rtol=1e-4, atol=1e-8,
                err_msg=f"Central FD disagrees with ForwardDiff (step={options.step:g})",
            )

            samples = {name: [] for name in functions}
            names = list(functions)
            for i in range(options.repeats):
                order = names if i % 2 == 0 else list(reversed(names))
                for name in order:
                    start = time.perf_counter()
                    functions[name]()
                    samples[name].append(1000 * (time.perf_counter() - start))

            ms = {name: statistics.median(data) for name, data in samples.items()}
            speedup = ms["finite_diff"] / ms["gtpsa_coeff"]
            row = dict(
                particles=n_particles,
                magnets=n_magnets,
                finite_diff_ms=ms["finite_diff"],
                forwarddiff_ms=ms["forwarddiff"],
                gtpsa_di_ms=ms["gtpsa_di"],
                gtpsa_coeff_ms=ms["gtpsa_coeff"],
                fd_over_gtpsa_coeff=speedup,
                finite_diff_max_abs_error=fd_error,
                fd_step=options.step,
            )
            rows.append(row)
            print(f"{n_particles:9d} {n_magnets:7d} "
                  f"{ms['finite_diff']:7.2f} {ms['forwarddiff']:11.2f} "
                  f"{ms['gtpsa_di']:12.2f} {ms['gtpsa_coeff']:15.2f} "
                  f"{speedup:8.2f} {fd_error:11.3e}", flush=True)

    if options.csv:
        with open(options.csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {options.csv}")


if __name__ == "__main__":
    main()
