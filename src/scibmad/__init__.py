"""SciBmad from Python.

``import scibmad as sb`` gives every name of the Julia package SciBmad (and of the packages it
re-exports: Beamlines, BeamTracking, GTPSA, NonlinearNormalForm, ...): ``sb.Quadrupole``,
``sb.Beamline``, ``sb.twiss``, ``sb.track``, ... Julia's ``f!`` is ``sb.f_`` (``sb.match`` and a
few other functions also have their name without the ``!``). Julia starts on first use.

What Python writes differently (all of it is described in the documentation, with the Julia
form next to the Python one):

* ``@elements`` is ``with sb.elements(): ...``; ``@knob(qf.Kn1)`` is ``sb.Knob(qf, "Kn1")``;
* indexing with keywords is a call: Julia's ``res.x[qd, turn=2]`` is ``res.x(qd, turn=2)``;
* ``end``/``begin`` in indexing are ``-1`` (or ``sb.end``, ``sb.end - 1``) and ``sb.begin``;
  ``:`` is ``:`` in brackets and ``...`` (or ``slice(None)``) elsewhere; an s-range ``(a, b)``
  is the slice ``a:b`` of floats;
* ``DefExpr(c -> -c.kq)`` is ``sb.DefExpr(lambda c: -c.kq)`` (traced into a Julia expression);
* ``withvalues(c; k=v) do ... end`` is ``with sb.withvalues(c, k=v): ...``;
* GPU arrays come from ``sb.device("cuda").asarray(...)`` (or "metal", "amdgpu", "oneapi").

Integers that number the elements of a beamline (beamline indexes, ``bl[1]``, ``tw.beta1[3]``)
and particles start at 1, as in Julia; NumPy arrays start at 0, as always in Python.
"""

from __future__ import annotations

from ._version import __version__
from ._runtime import SCIBMAD_VERSION, JuliaNotFoundError
from . import _core
from ._core import (JuliaArray, JuliaCallable, JuliaCollection, JuliaError, JuliaModule,
                    JuliaNumber, JuliaObject, Symbol, begin, end)
from ._extras import PerformanceWarning, elements, julia, withvalues
from ._gpu import Device, device

try:  # (so that `import juliacall` after `import scibmad` starts the same Julia)
    from . import _runtime as _rt
    _rt.prepare()
except Exception as _e:  # reported when Julia is started
    _rt.STATE["prepare_error"] = _e

# Python names of Julia functions named f! (also available as f_)
ALIASES = {
    "match": "match!",
    "undo": "undo!",
    "setknobs": "setknobs!",
    "phase_trombone": "phase_trombone!",
    "scalarize": "scalarize!",
}

_PYTHON_NAMES = {
    "__version__", "SCIBMAD_VERSION", "JuliaNotFoundError", "JuliaArray", "JuliaCallable",
    "JuliaCollection", "JuliaError", "JuliaModule", "JuliaNumber", "JuliaObject", "Symbol",
    "begin", "end", "PerformanceWarning", "elements", "julia", "withvalues", "Device",
    "device", "init", "info", "options", "DefExpr",
}


def init(julia=None, project=None, threads=None):
    """Starts Julia and loads SciBmad (otherwise done on the first use of a SciBmad name).

    julia:   a Julia executable (default: see ``scibmad.info()`` and the documentation)
    project: the Julia project in which SciBmad is installed (needed with `julia`, unless it is
             the SciBmad distribution's Julia)
    threads: the number of Julia threads (an int or "auto"); only before Julia has started
    """
    from . import _runtime
    _runtime.STATE.pop("prepare_error", None)
    _runtime.start(julia=julia, project=project, threads=threads)
    _core._init()


def info():
    """Which Julia and SciBmad.jl run SciBmad for Python."""
    from . import _runtime
    _core._init()
    return _runtime.info()


class _Options:
    """Display options: ``scibmad.options.display_width`` and ``display_height`` (characters;
    default: the terminal's size, 120 x 40 in notebooks)."""

    def __getattr__(self, name):
        try:
            return _core._OPTIONS[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        if name not in _core._OPTIONS:
            raise AttributeError(f"Unknown option {name!r}")
        _core._OPTIONS[name] = value

    def __repr__(self):
        return f"scibmad.options({_core._OPTIONS})"


options = _Options()

_DEFEXPR = None


def __getattr__(name):
    """The names of SciBmad (loaded on first use)."""
    if name.startswith("__"):
        raise AttributeError(name)
    _core._init()
    G, jl = _core.G, _core.jl
    if name == "DefExpr":
        global _DEFEXPR
        if _DEFEXPR is None:
            from ._extras import _DefExprType
            _DEFEXPR = _DefExprType.__new__(_DefExprType)
            object.__setattr__(_DEFEXPR, "_jl", G.getname(jl.SciBmad, jl.Symbol("DefExpr")))
        return _DEFEXPR
    jname = ALIASES.get(name)
    s = jl.Symbol(jname) if jname is not None else G.jlname(jl.SciBmad, name)
    if s is None:
        raise AttributeError(f"SciBmad has no name {name!r}")
    v = _core.wrap(_core._jcall(G.getname, jl.SciBmad, s))
    if isinstance(v, (JuliaCallable, JuliaModule)):
        globals()[name] = v  # (functions, types and modules do not change)
    return v


def __dir__():
    names = set(_PYTHON_NAMES) | set(ALIASES)
    try:
        _core._init()
        names |= {str(n) for n in _core.G.pynames(_core.jl.SciBmad) if str(n).isidentifier()}
    except Exception:
        pass
    return sorted(names)
