"""The parts of the Python interface whose form differs from Julia's: what Julia writes with
macros (``@elements``, ``@knob``), ``do`` blocks (``withvalues``) or Julia closures
(``DefExpr(c -> c.kq)``)."""

from __future__ import annotations

import inspect
import sys
import warnings

from . import _core
from ._core import JuliaCallable, JuliaNumber, JuliaObject, _jcall, to_julia, wrap


class PerformanceWarning(UserWarning):
    """A Python function is called from Julia where Julia code would be much faster."""


# ---------------------------------------------------------------------------------------------
# Deferred expressions


class _TraceVar(JuliaNumber):
    """A context variable while a Python function is traced into a DefExpr: indexing it gives
    an entry (counted from 0, as the NumPy array the variable is in Python)."""

    __slots__ = ()

    def __getitem__(self, i):
        import numpy as np
        if not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)):
            raise _core._TracingAbort
        if i < 0:
            raise _core._TracingAbort  # (the length is only known when it is evaluated)
        return wrap(_core.G.ctxindex(self._jl, int(i) + 1))

    def __getattr__(self, name):
        raise _core._TracingAbort


class _TraceContext:
    __slots__ = ()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = _core.G.ctxvar(name)
        p = _TraceVar.__new__(_TraceVar)
        object.__setattr__(p, "_jl", v)
        return p

    __getitem__ = __getattr__


def _nparams(f):
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return 1
    ps = [p for p in sig.parameters.values()
          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    return len(ps)


def _trace(f):
    """The Julia DefExpr that the Python function f builds from the context variables, or None
    if f does something with them that only Python can do (then it is called from Julia)."""
    n = _nparams(f)
    if n > 1:
        return None
    _core._TRACING[0] = True
    try:
        r = f(_TraceContext()) if n == 1 else f()
    except _core._TracingAbort:
        return None
    except Exception:
        return None
    finally:
        _core._TRACING[0] = False
    if isinstance(r, JuliaObject) and _core.G.isdefexpr(r._jl):
        return r._jl
    if isinstance(r, (int, float, complex)) or isinstance(r, JuliaNumber):
        return _core.G.ctxconst(to_julia(r))
    return None


_DEFEXPR_DOC = """``DefExpr(f)``: a deferred expression. ``f`` is a function of the context
    (``lambda c: -c.kq``) or without arguments. It is called once, with symbolic context
    variables, and the Julia expression that it builds from them is what is evaluated later:
    as fast as a Julia function, with nothing of Python left in it. A function that does more
    with the variables than arithmetic and NumPy functions (e.g. ``math.sin``, an ``if`` on
    them) is called from Julia instead each time the expression is evaluated, which is much
    slower and works only on the CPU (a :class:`PerformanceWarning` says so).

    Python values that ``f`` uses (not context variables) are read when ``DefExpr`` is called.
"""


class _DefExprType(JuliaCallable):
    __slots__ = ()

    def __call__(self, f, *args, **kwargs):
        if callable(f) and not isinstance(f, JuliaObject) and not args and not kwargs:
            d = _trace(f)
            if d is not None:
                return wrap(d)
            warnings.warn("This DefExpr calls its Python function every time it is evaluated "
                          "(it uses the context variables in a way that cannot be traced into "
                          "Julia); it is slower, and works only on the CPU", PerformanceWarning,
                          stacklevel=2)
        return JuliaCallable.__call__(self, f, *args, **kwargs)

    @property
    def __doc__(self):
        return _DEFEXPR_DOC + "\n\nJulia documentation:\n\n" + \
            (JuliaCallable.__doc__.fget(self) or "")


# ---------------------------------------------------------------------------------------------
# @elements


def _is_element(v):
    return isinstance(v, JuliaObject) and _core.G.iselement(v._jl)


class elements:
    """Names elements after the variables they are assigned to, as Julia's ``@elements``::

        with scibmad.elements():
            qf = Quadrupole(Kn1=0.36, L=0.5)
            d = Drift(L=0.1)
        qf.name  # "qf"

    Every variable that the block assigns an element to gets its name. ``elements(namespace)``
    names the elements of a dictionary (e.g. ``elements(globals())``) by their keys, if they
    have no name yet. (Elements can also be named when they are made: ``Drift(name="d")``.)
    """

    def __init__(self, namespace=None):
        self._namespace = namespace
        self._before = None
        if namespace is not None:
            for k, v in list(namespace.items()):
                if isinstance(k, str) and not k.startswith("_") and _is_element(v) and \
                        not str(v.name):
                    v.name = k

    def __enter__(self):
        frame = sys._getframe(1)
        self._before = {k: id(v) for k, v in frame.f_locals.items()}
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        frame = sys._getframe(1)
        for k, v in list(frame.f_locals.items()):
            if self._before.get(k) != id(v) and not k.startswith("_") and _is_element(v):
                v.name = k
        return False


# ---------------------------------------------------------------------------------------------
# withvalues


class _WithValues:
    __slots__ = ("_ctx", "_values", "_n")

    def __init__(self, ctx, values):
        self._ctx = ctx
        self._values = values
        self._n = None

    def __enter__(self):
        c = to_julia(self._ctx)
        self._n = int(_core.G.nsaved(c))
        _jcall(_core.G.getname(_core.jl.Base, _core.jl.Symbol("push!")), c,
               **_core._kwargs_to_julia(self._values))
        return self._ctx

    def __exit__(self, *exc):
        _jcall(_core.G.popto, to_julia(self._ctx), self._n)
        return False


def withvalues(*args, **values):
    """Temporary values of context variables, as Julia's ``withvalues(f, c; kwargs...)``::

        with scibmad.withvalues(ctx, vrf=2e6):
            ...  # ctx.vrf is 2e6 here
        # and its old value again here (also if the block raised an exception)

    ``withvalues(f, ctx, **values)`` calls ``f()`` with the values set and returns its result.
    """
    _core._init()
    if len(args) == 2 and callable(args[0]):
        f, ctx = args
        jf = _core.G.getname(_core.jl.SciBmad, _core.jl.Symbol("withvalues"))
        return wrap(_jcall(jf, to_julia(f), to_julia(ctx), **_core._kwargs_to_julia(values)))
    if len(args) != 1:
        raise TypeError("withvalues(ctx, **values) or withvalues(f, ctx, **values)")
    return _WithValues(args[0], values)


# ---------------------------------------------------------------------------------------------
# The Julia session


class _Julia:
    """The Julia session that runs SciBmad.

    * ``julia.Main``, ``julia.Base``, ``julia.SciBmad``: Julia modules (their names are
      attributes; ``f!`` is ``f_``; ``module["name"]`` gives any name, e.g. an operator);
    * ``julia.package("Optim")``: loads a Julia package (``install=True`` installs it first if
      needed);
    * ``julia.install("Optim")``: installs a Julia package into the environment SciBmad runs in;
    * ``julia.include("lattice.jl")``: runs a Julia file in a module of its own and returns it;
    * ``julia.seval(code)``: evaluates Julia code (never needed for SciBmad itself).
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        _core._init()
        if name in ("Main", "Base", "Core"):
            return wrap(getattr(_core.jl, name))
        mod = _core.G.getname(_core.jl.Main, _core.jl.Symbol(name)) \
            if bool(_core.jl.isdefined(_core.jl.Main, _core.jl.Symbol(name))) else None
        if mod is None:
            raise AttributeError(f"Julia has no module {name!r} loaded; use "
                                 f"julia.package({name!r})")
        return wrap(mod)

    def package(self, name, install=False):
        """Loads the Julia package `name` and returns it (installing it first if `install`)."""
        _core._init()
        if not bool(_core.G.isinstalled(name)):
            if not install:
                raise ModuleNotFoundError(f"The Julia package {name} is not installed; use "
                                          f"scibmad.julia.install({name!r})")
            self.install(name)
        return wrap(_jcall(_core.G.load, name))

    def install(self, name):
        """Installs the Julia package `name` into the environment SciBmad runs in."""
        _core._init()
        info = _core._runtime.STATE["info"] or {}
        print(f"scibmad: installing the Julia package {name} (once; this may take a while)",
              file=sys.stderr)
        _jcall(_core.G.install, name)
        if info.get("kind") == "juliapkg":
            # (recorded with juliapkg too, which otherwise removes it when it resolves again)
            import juliapkg
            juliapkg.add(name, str(_core.G.pkguuid(name)))

    def include(self, path):
        """Runs the Julia file `path` (e.g. a lattice file) in a module of its own, which uses
        SciBmad, and returns that module: the variables the file defines are its attributes
        (`lat = sb.julia.include("esr.jl"); lat.ring`)."""
        _core._init()
        import os
        return wrap(_jcall(_core.G.include_file, os.fspath(path)))

    def seval(self, code):
        """Evaluates Julia code in Main and returns the result."""
        _core._init()
        return wrap(_jcall(_core.jl.seval, code))

    def __repr__(self):
        return "<the Julia session of scibmad>"


julia = _Julia()
