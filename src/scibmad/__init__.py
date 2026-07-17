"""Python interface to the SciBmad.jl accelerator physics ecosystem.

Basic usage::

    import scibmad as sb

    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
    d = sb.Drift(L=1.0)

    bl = sb.Beamline([qf, d, qd, d], species_ref=sb.Species("electron"),
                     E_ref=18e9)

    tw = sb.twiss(bl)

Any exported name of the ``SciBmad`` Julia module (element types like
``Quadrupole``/``Sextupole``/``SBend``, functions like ``track``,
``find_closed_orbit``, constants, ...) is available as ``scibmad.<name>`` even if
it is not listed explicitly below -- it is looked up lazily and wrapped so that
``scibmad``'s deferred-parameter operands are unwrapped on the way in and
re-wrapped on the way out.
"""

from __future__ import annotations

# Bootstrapping the runtime (and thus Julia + SciBmad) happens on import.
from ._runtime import LOOKUP_MODULES, SciBmad, jl
from ._wrappers import Operand, unwrap, wrap
from ._api import (
    BatchParam,
    Beamline,
    DefExpr,
    MATH_FUNCTIONS,
    Time,
    TimeDependentParam,
    elements,
    name,
    twiss,
)

# Expose the top-level math functions (sin, cos, sqrt, ...) that understand both
# plain numbers and scibmad operands.
globals().update(MATH_FUNCTIONS)

# Names defined in this module (so __getattr__ is only a fallback for SciBmad
# pass-through) and the curated public surface.
_EXPLICIT = {
    "jl",
    "SciBmad",
    "Operand",
    "unwrap",
    "wrap",
    "DefExpr",
    "TimeDependentParam",
    "BatchParam",
    "Time",
    "Beamline",
    "twiss",
    "elements",
    "name",
    *MATH_FUNCTIONS.keys(),
}

__all__ = sorted(_EXPLICIT)


def _make_passthrough(jl_callable):
    """Wrap a Julia callable so Python operands are unwrapped in and results
    (re)wrapped out."""

    def _call(*args, **kwargs):
        a = [unwrap(x) for x in args]
        k = {key: unwrap(v) for key, v in kwargs.items()}
        return wrap(jl_callable(*a, **k))

    return _call


def __getattr__(attr):
    """Lazily surface any exported SciBmad name (PEP 562).

    Callables (functions, element/type constructors) are wrapped to handle
    operand (un)wrapping; other values are returned as juliacall gives them.
    """
    if attr.startswith("__") and attr.endswith("__"):
        raise AttributeError(attr)
    # Try SciBmad first, then the subpackages -- this resolves names that are
    # ambiguous at the reexport level (e.g. Context) to the accelerator-facing one.
    _MISSING = object()
    obj = _MISSING
    for module in LOOKUP_MODULES:
        try:
            obj = getattr(module, attr)
            break
        except Exception:  # noqa: BLE001 - ambiguous/undefined here; try next module
            continue
    if obj is _MISSING:
        raise AttributeError(
            f"module 'scibmad' has no attribute {attr!r} "
            f"(no such export in SciBmad or its subpackages)"
        )
    if callable(obj):
        return _make_passthrough(obj)
    return obj


def __dir__():
    extra = [str(n) for n in jl.names(SciBmad)]
    return sorted(_EXPLICIT.union(extra))
