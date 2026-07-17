"""High-level Python API for SciBmad.

This module defines the pieces of the surface that need Python-specific handling
(constructing deferred types from Python callables, the ``at=`` argument of
:func:`twiss`, the ``@elements``-style helper). Everything else is surfaced
automatically by :func:`scibmad.__getattr__`.
"""

from __future__ import annotations

import inspect
import types

from ._runtime import SciBmad, UNARY_OP, glue, jl
from ._wrappers import Operand, unwrap, wrap

__all__ = [
    "DefExpr",
    "TimeDependentParam",
    "BatchParam",
    "Time",
    "Beamline",
    "twiss",
    "elements",
    "name",
]

# --- dtype handling ----------------------------------------------------------

_DTYPE_MAP = {
    float: jl.Float64,
    int: jl.Int64,
    complex: jl.ComplexF64,
    bool: jl.Bool,
}


def _julia_type(dtype):
    """Map a Python dtype (or a Julia type) to the Julia ``DataType`` to use."""
    if dtype is None:
        return jl.Float64
    if dtype in _DTYPE_MAP:
        return _DTYPE_MAP[dtype]
    # Assume it is already a Julia type object.
    return dtype


def _positional_arity(f):
    """Number of required positional parameters of a Python callable.

    Used to tell a 0-argument deferred expression (``lambda: ...``) from a
    Context/time-accepting one (``lambda c: ...``). Falls back to 1 when the
    signature cannot be introspected.
    """
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return 1
    n = 0
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and p.default is inspect.Parameter.empty:
            n += 1
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            return 1
    return n


# --- deferred-type constructors ----------------------------------------------

def DefExpr(f, dtype=float):
    """Build a lazily-evaluated deferred expression from a Python callable or value.

    ``f`` may be:

    * a 0-argument callable, e.g. ``sb.DefExpr(lambda: a)``;
    * a 1-argument callable receiving a ``Context``, e.g.
      ``sb.DefExpr(lambda c: c.k1)``;
    * a plain number, e.g. ``sb.DefExpr(0.36)``.

    ``dtype`` sets the value type of the expression (default ``float`` ->
    ``Float64``). The result supports Python arithmetic::

        sb.DefExpr(lambda: 1) + 10
        sb.DefExpr(lambda: 1) + sb.DefExpr(lambda: 2)
    """
    T = _julia_type(dtype)
    if isinstance(f, Operand):
        f = f._jl
    if callable(f):
        if _positional_arity(f) == 0:
            return Operand(glue.defexpr_from_py0(f, T))
        return Operand(glue.defexpr_from_py1(f, T))
    # A plain value: wrap it in a 0-argument expression.
    value = f
    return Operand(glue.defexpr_from_py0(lambda: value, T))


def TimeDependentParam(f, dtype=float):
    """Build a time-dependent parameter from a Python callable ``f(t)`` or a value.

    Supports Python arithmetic, mirroring :func:`DefExpr`.
    """
    T = _julia_type(dtype)
    if isinstance(f, Operand):
        f = f._jl
    if callable(f):
        return Operand(glue.tdparam_from_py(f, T))
    value = f
    return Operand(glue.tdparam_from_py(lambda t: value, T))


def Time():
    """The identity time-dependent parameter ``t -> t`` (SciBmad's ``Time()``)."""
    return Operand(SciBmad.Time())


def BatchParam(batch, dtype=float):
    """Build a batch parameter from a Python sequence (or a scalar).

    Batches let particles see differing parameter values; see SciBmad's
    ``BatchParam``. Supports Python arithmetic.
    """
    if isinstance(batch, Operand):
        batch = batch._jl
    T = _julia_type(dtype)
    if isinstance(batch, (list, tuple)):
        return Operand(glue.batchparam_from_py([unwrap(x) for x in batch], T))
    return Operand(glue.batchparam_scalar(unwrap(batch), T))


# --- Beamline ----------------------------------------------------------------

def Beamline(line, **kwargs):
    """Construct a ``Beamline`` from a Python list of elements.

    The element list is converted to a genuine Julia vector first, so a plain
    Python list works::

        bl = sb.Beamline([qf, d, qd, d], species_ref=sb.Species("electron"),
                         E_ref=18e9)
    """
    jl_line = glue.to_any_vector([unwrap(e) for e in line])
    jl_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
    return wrap(SciBmad.Beamline(jl_line, **jl_kwargs))


# --- twiss -------------------------------------------------------------------

def twiss(bl, at=None, **kwargs):
    """Compute Twiss parameters for a beamline.

    ``at`` accepts a Python list of element indices, ``LineElement`` objects
    and/or ``(s0, s1)`` ranges (it is converted to a Julia vector so it works
    without SciBmad depending on PythonCall). Omit ``at`` for the default
    (every element)::

        tw = sb.twiss(bl)
        tw = sb.twiss(bl, at=[])
        tw = sb.twiss(bl, spin=True)
    """
    jl_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
    if at is not None:
        jl_kwargs["at"] = glue.to_any_vector([unwrap(x) for x in at])
    return wrap(SciBmad.twiss(unwrap(bl), **jl_kwargs))


# --- @elements equivalent ----------------------------------------------------

def name(ele, ele_name):
    """Set an element's ``name`` and return it (mirrors Julia's ``setname``)::

        qf = sb.name(sb.Quadrupole(L=0.5), "qf")
    """
    target = unwrap(ele)
    target.name = ele_name
    return ele


def elements(mapping=None, /, **kwargs):
    """Python equivalent of Julia's ``@elements`` macro.

    Give each element a name equal to its key and return a namespace whose
    attributes are those named elements::

        E = sb.elements(
            qf=sb.Quadrupole(Kn1=0.36, L=0.5),
            d=sb.Drift(L=1.0),
            qd=sb.Quadrupole(Kn1=-0.36, L=0.5),
        )
        E.qf.name            # -> "qf"
        bl = sb.Beamline([E.qf, E.d, E.qd, E.d], ...)

    A dict may be passed positionally instead of/along with keywords. Only
    elements whose name is still unset are (re)named, matching ``@elements``.
    """
    items = {}
    if mapping is not None:
        items.update(dict(mapping))
    items.update(kwargs)

    ns = types.SimpleNamespace()
    for ele_name, ele in items.items():
        target = unwrap(ele)
        try:
            if str(target.name) == "":
                target.name = ele_name
        except Exception:
            pass
        setattr(ns, ele_name, ele)
    return ns


# --- top-level math functions ------------------------------------------------

def _make_unary(jl_fn):
    def _fn(x):
        return wrap(jl_fn(unwrap(x)))
    return _fn


# Populated into the package namespace by __init__ (sin, cos, sqrt, ...).
MATH_FUNCTIONS = {n: _make_unary(fn) for n, fn in UNARY_OP.items()}
