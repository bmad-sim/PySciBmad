# PySciBmad

`scibmad` is the Python interface to the
[SciBmad.jl](https://github.com/bmad-sim/SciBmad.jl) accelerator physics
simulation ecosystem. It exposes the full SciBmad feature set to Python through
[`juliacall`](https://github.com/JuliaPy/PythonCall.jl), while adding the
Python-specific ergonomics needed to make it feel native.

> The distributed package is named **`scibmad`** (`import scibmad as sb`); this
> repository is `PySciBmad`.

📄 **Browsable docs:** a self-contained HTML version of this documentation lives at
[`docs/index.html`](https://github.com/bmad-sim/PySciBmad/blob/main/docs/index.html)
— open it directly in a browser.

## Installation

```bash
pip install scibmad
```

A conda-forge package is planned but not yet available.

The first import installs, via `juliapkg`, the **long-term-support (LTS)** release
of Julia (the `1.10` series — *not* the latest Julia) together with the
registered `SciBmad.jl`. This is pinned in
[`src/scibmad/juliapkg.json`](https://github.com/bmad-sim/PySciBmad/blob/main/src/scibmad/juliapkg.json).
Updating the Python package (e.g. `pip install -U scibmad`) updates the pinned
Julia/SciBmad versions along with it.

## Quick start

```python
import scibmad as sb

qf = sb.Quadrupole(Kn1=0.36, L=0.5)
qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
d = sb.Drift(L=1.0)

bl = sb.Beamline(
    [qf, d, qd, d],
    species_ref=sb.Species("electron"),
    E_ref=18e9,
)

tw = sb.twiss(bl)          # Twiss at every element
tw = sb.twiss(bl, at=[])   # `at` accepts a Python list
```

## Deferred / interdependent parameters

Deferred expressions, time-dependent parameters and batch parameters all work
from Python, **including mixed arithmetic with Python numbers** — which plain
`juliacall` cannot provide for these types (see *How it works* below for why):

```python
k = sb.DefExpr(lambda c: c.k1)     # closes over a Context field
qf = sb.Quadrupole(Kn1=k, L=0.5)
qd = sb.Quadrupole(Kn1=-k, L=0.5)  # unary minus works

sb.DefExpr(lambda: 1) + 10                 # -> DefExpr
sb.DefExpr(lambda: 1) + sb.DefExpr(lambda: 2)
10 * sb.DefExpr(lambda: 2)                  # reflected operators too

t = sb.Time()                              # identity time parameter
ramp = 1e6 * sb.sin(2 * 3.14159 * t)       # arithmetic + math functions

b = sb.BatchParam([0.1, 0.2, 0.3])         # per-particle values
```

A deferred value can be evaluated by calling it:

```python
e = sb.DefExpr(lambda: 1) + 2
e()          # -> 3.0
```

> **Known limitations (upstream, fix pending).** Two `DefExpr` bugs in Beamlines
> 0.9.4, both fixed by
> [Beamlines.jl#161](https://github.com/bmad-sim/Beamlines.jl/pull/161). This
> note comes out once that is merged and released.
>
> 1. **`abs` is missing on `DefExpr`.** `abs(sb.DefExpr(lambda: -1.0))` and
>    `sb.abs(k)` raise a `MethodError` — `Base.abs` is simply not defined for
>    `DefExpr`, unlike every other single-argument math function. This affects
>    *all* `DefExpr`s, including zero-argument ones. `TimeDependentParam` and
>    `BatchParam` are unaffected; they do define it.
> 2. **Operators drop an explicitly supplied `Context`.** For a `DefExpr` built
>    from a *Context-accepting* lambda, `(-k)(ctx)` and `(k + 1)(ctx)` evaluate
>    `k` against the null context instead of `ctx`, so they either raise
>    "Variable ... is not defined in the local Context" or, where the name
>    resolves via `GLOBAL_CONTEXTS`, quietly return the wrong value. The same
>    applies to a `Beamline`'s stored context reaching an operator-built
>    parameter. Zero-argument `DefExpr`s, `TimeDependentParam` and `BatchParam`
>    are unaffected.

## Naming elements (`@elements` equivalent)

Julia's `@elements` macro sets each element's `name` to its variable name. The
Python equivalent is `sb.elements(...)`, which names each element after its key
and returns a namespace:

```python
E = sb.elements(
    qf=sb.Quadrupole(Kn1=0.36, L=0.5),
    d=sb.Drift(L=1.0),
    qd=sb.Quadrupole(Kn1=-0.36, L=0.5),
)
E.qf.name          # -> "qf"
bl = sb.Beamline([E.qf, E.d, E.qd, E.d], species_ref=sb.Species("electron"),
                 E_ref=18e9)
```

For a single element, `sb.name(ele, "qf")` sets and returns it.

## Accessing the full SciBmad surface

Any exported name of the `SciBmad` Julia module is available as `scibmad.<name>`
even if it is not wrapped explicitly — element types (`Sextupole`, `SBend`,
`Solenoid`, `RFCavity`, `Marker`, `Multipole`, ...), functions (`track`,
`find_closed_orbit`, `dynamic_aperture`, ...) and constants. For advanced use the
raw Julia session and SciBmad module are exposed:

```python
sb.jl          # juliacall Main
sb.SciBmad     # the SciBmad Julia module
sb.Operand     # the operator-overloading wrapper type
sb.unwrap, sb.wrap
```

## How it works / relationship to SciBmad.jl

* **Thin `juliacall` bridge.** Element/beamline construction, tracking and Twiss
  are forwarded to `SciBmad.jl`; Python operands are unwrapped on the way in and
  re-wrapped on the way out.
* **Operator overloading fix.** `scibmad.Operand` forwards Python arithmetic to
  the Julia operators defined on `DefExpr` / `TimeDependentParam` / `BatchParam`.
* **Deferred-type construction from Python callables** and the `at=`
  Python-list handling live in
  [`src/scibmad/_glue.jl`](https://github.com/bmad-sim/PySciBmad/blob/main/src/scibmad/_glue.jl),
  loaded at import time. This keeps the Python package working against the
  *registered* Julia packages today.

### Recommended upstream (Julia) changes

Per [SciBmad.jl#76](https://github.com/bmad-sim/SciBmad.jl/issues/76), the glue
should eventually move into `PythonCall` package extensions in the relevant
SciBmad subpackages (so no package depends on `PythonCall` directly), at which
point the corresponding glue here can be dropped. Each piece is up as a **draft
PR**; none is merged or released yet, so every glue function below is still
required:

| Upstream change | PR | Glue it would retire |
| --- | --- | --- |
| Generalize `twiss(...; at::Union{Colon, Vector})` to `AbstractVector`, so a `PyArray`/`PyList` works without conversion | [SciBmad.jl#81](https://github.com/bmad-sim/SciBmad.jl/pull/81) | `to_any_vector` in the `at=` path of `twiss` |
| `DefExpr(::Py)` construction from Python callables, in `BeamlinesPythonCallExt` | [Beamlines.jl#158](https://github.com/bmad-sim/Beamlines.jl/pull/158) | `defexpr_from_py0` / `defexpr_from_py1` |
| New `BeamTrackingPythonCallExt` for `TimeDependentParam(::Py)` / `BatchParam(::Py)` | [BeamTracking.jl#315](https://github.com/bmad-sim/BeamTracking.jl/pull/315) | `tdparam_from_py`, `batchparam_from_py`, `batchparam_scalar` |

`Beamline`'s own list handling already landed upstream — as of Beamlines 0.9.4
its constructor does `convert(Vector{LineElement}, vec(line))`, so the
`to_any_vector` call in `sb.Beamline` is belt-and-braces rather than required.

The operator-overloading wrappers in `_wrappers.py` are *not* on this list: they
compensate for a `juliacall` limitation, not for anything missing in SciBmad.

The limitation is narrower than "no operators". `juliacall` defines the
arithmetic dunders on `AnyValue`, the wrapper for *every* Julia value, so
`DefExpr + DefExpr` and `-DefExpr` already work without help. But the underlying
`pyjlany_op` returns `NotImplemented` unless the *other* operand is also a Julia
value; only `Number` subtypes get `pyjlnumber_op`, which additionally
`pyconvert`s a Python number. So it is specifically **mixed arithmetic with a
Python scalar** — `DefExpr(...) + 10`, `10 * DefExpr(...)` — that raises
`TypeError` without `Operand`. The clean upstream fix would be in PythonCall
(give `pyjlany_op` the same convert fallback), not in SciBmad.

### Verified against

The suite passes end-to-end against the current registered stack: Julia 1.10.11
(LTS), SciBmad 0.4.2, Beamlines 0.9.4, BeamTracking 0.7.1,
AtomicAndPhysicalConstants 0.10.0, PythonCall 0.9.35.

## Development / testing

```bash
pip install -e ".[test]"
pytest
```
