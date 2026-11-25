# PySciBmad

SciBmad from Python. The package `pyscibmad` (`import scibmad as sb`) runs
[SciBmad.jl](https://github.com/bmad-sim/SciBmad.jl), the Julia toolkit for charged particle
beam dynamics, inside Python: every function and type of SciBmad is a name of `scibmad`, with
the same arguments, and the results are Python objects (NumPy arrays where they are numbers).

```python
import numpy as np
import matplotlib.pyplot as plt
import scibmad as sb

with sb.elements():                      # names the elements after their variables
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    d = sb.Drift(L=1.0)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
fodo = sb.Beamline([qf, d, qd, d], species_ref=sb.Species("electron"), pc_ref=18e9)

tw = sb.twiss(fodo)
tw.q1, tw.beta1[qd]                      # a summary value, a column at an element
plt.plot(tw.s, tw.beta1)

te = sb.TwissExpr(fodo)                  # match the tunes, with exact derivatives
sb.match(vary=[sb.Knob(qf, "Kn1"), sb.Knob(qd, "Kn1")],
         targets=[te.q1 == 0.2, te.q2 == 0.2])

res = sb.track(fodo, v0=np.random.rand(1000, 6) * 1e-4, n_turns=100)
res.x(turn=100)                          # Julia's res.x[turn=100]
```

The [SciBmad documentation](https://bmad-sim.github.io/SciBmad.jl) shows every example in
Python or in Julia (the switch at the top of each page). What Python writes differently from
Julia, and why, is on its page
[SciBmad from Python](https://bmad-sim.github.io/SciBmad.jl/python/python.html).

## Installation

```bash
conda install -c bmad-sim -c conda-forge pyscibmad   # with the precompiled SciBmad distribution
pip install pyscibmad                                 # or: Julia and SciBmad are installed on first use
```

There is nothing else to install: Julia runs inside the Python process. Which Julia runs SciBmad
is decided at `import scibmad`, in this order:

1. `sb.init(julia=..., project=..., threads=...)` before the first use of SciBmad;
2. the environment variables `SCIBMAD_JULIA` (a Julia executable) and `SCIBMAD_PROJECT` (a
   Julia project with SciBmad and PythonCall), and `SCIBMAD_THREADS`;
3. the configuration of the user, set with `python -m scibmad config --julia ... --project ...`
   (`--reset` goes back to the default; `python -m scibmad info` shows what runs);
4. juliacall's own `PYTHON_JULIACALL_EXE` and `PYTHON_JULIACALL_PROJECT`, if set;
5. the SciBmad distribution of the current conda environment (`conda install -c bmad-sim
   scibmad`, a Julia with SciBmad and the packages it uses precompiled);
6. otherwise [juliapkg](https://github.com/JuliaPy/pyjuliapkg) installs Julia and the SciBmad.jl
   version of this release (`src/scibmad/juliapkg.json`) into the Python environment, once
   (several minutes, for the precompilation).

Each release of `pyscibmad` is made for one release of SciBmad.jl (`scibmad.SCIBMAD_VERSION`):
updating the Python package updates SciBmad with it (`pip install -U pyscibmad`, `conda update
pyscibmad`). Other Julia installations are checked against the supported SciBmad versions (a
warning if outside).

GPUs: `sb.device("cuda")` (or `"metal"`, `"amdgpu"`, `"oneapi"`) installs the Julia package of
that GPU the first time (also `python -m scibmad gpu cuda`) and gives `asarray`, which copies
NumPy arrays to it.

## How it works

The package is a thin, generic layer on [juliacall](https://github.com/JuliaPy/PythonCall.jl):
nothing in it lists SciBmad's API, so new functionality of SciBmad.jl is available in Python as
soon as SciBmad has it.

- `src/scibmad/_runtime.py`: which Julia runs SciBmad, and starting it. For the SciBmad
  distribution and for a given Julia project, the `juliacall` Python package that comes with
  their PythonCall is used, so that the two versions always match (juliacall requires it).
  Julia's signal handlers are on (Julia's garbage collector needs them as soon as it has more
  than one thread, e.g. those of GPU packages), with Python's own handler for Ctrl-C.
- `src/scibmad/_core.py`: the proxies of Julia values (attributes, indexing, calls, operators,
  display, NumPy and Matplotlib) and the conversions between Python and Julia values.
- `src/scibmad/_glue.jl`: the Julia side (a module `PySciBmad` in `Main`): categories of values,
  indexing rules, calls with strings for Symbols, display for Python, errors.
- `src/scibmad/_extras.py`: what Python writes differently: `elements`, `withvalues`,
  `DefExpr` (traced into Julia expressions), `julia` (the Julia session: modules, packages,
  `include` of Julia lattice files).
- `src/scibmad/_gpu.py`: `device`.
- `src/scibmad/__init__.py`: the names of SciBmad as attributes of the module (`f!` is `f_`;
  `match`, `undo`, `setknobs`, `phase_trombone`, `scalarize` also without the underscore).

## Development

With your own Julia environment, e.g. one that develops the SciBmad packages:

```bash
julia --project=myenv -e 'using Pkg; Pkg.develop(path="path/to/SciBmad"); Pkg.add("PythonCall")'
pip install -e ".[test]"
SCIBMAD_JULIA=$(which julia) SCIBMAD_PROJECT=myenv pytest
```

The tests compare each feature with the same computation done in Julia. The examples of the
SciBmad documentation are the larger test: every Python cell of its pages runs (see the
SciBmad repository's `docs/_ext/dual_language.py`).

## License

GPL-3.0-or-later.
