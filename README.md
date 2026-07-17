# PySciBmad

`scibmad` is the Python interface to the
[SciBmad.jl](https://github.com/bmad-sim/SciBmad.jl) accelerator physics
simulation ecosystem. It exposes the full SciBmad feature set to Python through
[`juliacall`](https://github.com/JuliaPy/PythonCall.jl), while adding the
Python-specific ergonomics needed to make it feel native.

> The distributed package is named **`scibmad`** (`import scibmad as sb`); this
> repository is `PySciBmad`.

📄 **Browsable docs:** a self-contained HTML version of this documentation lives at
[`docs/index.html`](docs/index.html) — open it directly in a browser.

## Installation

```bash
pip install scibmad          # or: conda install scibmad
```

The first import installs, via `juliapkg`, the **long-term-support (LTS)** release
of Julia (the `1.10` series — *not* the latest Julia) together with the
registered `SciBmad.jl`. This is pinned in
[`src/scibmad/juliapkg.json`](src/scibmad/juliapkg.json). Updating the Python
package (e.g. `conda update scibmad`) updates the pinned Julia/SciBmad versions
along with it.

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
from Python, **including operator overloading** — which plain `juliacall` cannot
provide because these Julia types are not `Number` subtypes:

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
  Python-list handling live in [`src/scibmad/_glue.jl`](src/scibmad/_glue.jl),
  loaded at import time. This keeps the Python package working against the
  *registered* Julia packages today.

### Recommended upstream (Julia) changes

Per [SciBmad.jl#76](https://github.com/bmad-sim/SciBmad.jl/issues/76), the
following should eventually move into `PythonCall` package extensions in the
relevant SciBmad subpackages (so no package depends on `PythonCall` directly),
at which point the corresponding glue here can be dropped:

* Generalize `twiss(...; at::Union{Colon, Vector})` to accept `AbstractVector`
  (so a `PyArray`/`PyList` works without conversion).
* Move the `DefExpr` / `TimeDependentParam` / `BatchParam` construction-from-
  `Py` helpers into `BeamlinesPythonCallExt` / a `BeamTracking` PythonCall
  extension.

## Development / testing

```bash
pip install -e ".[test]"
pytest
```
