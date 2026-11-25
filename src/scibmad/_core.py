"""Python proxies of Julia values and the conversions between Python and Julia.

Every Julia value that has no natural Python equivalent is returned as a proxy, whose
attributes, indexing, calls, operators and display are those of the Julia value:

* attributes are Julia properties (``ele.Kn1``, ``tw.beta1``), also for setting them;
* ``obj[...]`` is Julia indexing, ``obj(...)`` a call, or indexing with keywords when the
  object cannot be called (``res.x(qd, turn=2)`` is Julia's ``res.x[qd, turn=2]``);
* operators are Julia's (``+``, ``*``, ``==``, ``<=``, ...), so expressions, deferred
  expressions, time-dependent and batch parameters and Taylor series combine as in Julia;
* ``repr`` is Julia's ``show``.

Julia arrays of plain numbers are returned as NumPy arrays (sharing memory with Julia when they
are CPU arrays); other Julia arrays as :class:`JuliaArray`, which indexes like NumPy (from 0).
"""

from __future__ import annotations

import re
import shutil
import sys
import weakref
from fractions import Fraction

import numpy as np

from . import _runtime

# Filled by _init() once Julia runs
jl = None  # juliacall's Main
G = None  # the glue module PySciBmad (see _glue.jl)
_ValueBase = None  # juliacall.ValueBase
_JuliaCallError = None  # juliacall.JuliaError
_F: dict = {}  # Julia functions used by the proxies


def _init():
    global jl, G, _ValueBase, _JuliaCallError
    if G is not None:
        return
    jl = _runtime.start()
    import juliacall

    _ValueBase = juliacall.ValueBase
    _JuliaCallError = juliacall.JuliaError
    G = jl.PySciBmad
    base = jl.Base
    for name in ("+", "-", "*", "/", "^", "==", "!=", "<", "<=", ">", ">=", "&", "|", "xor",
                 "mod", "fld", "~", "abs", "conj", "real", "imag", "hash", "copy", "deepcopy",
                 "length", "in", "isa", "broadcast", "Float64", "ComplexF64", "Int"):
        _F[name] = G.getname(base, jl.Symbol(name))
    _setup_ipython()


def julia_main():
    _init()
    return jl


# ---------------------------------------------------------------------------------------------
# Errors


class JuliaError(Exception):
    """An exception raised in Julia. ``julia_type`` is the name of its Julia type and
    ``julia_exception`` the Julia exception itself."""

    julia_type = None
    julia_exception = None


_ERROR_CLASSES: dict = {}


def _error_class(kind):
    cls = _ERROR_CLASSES.get(kind)
    if cls is None:
        builtin = {"IndexError": IndexError, "KeyError": KeyError, "TypeError": TypeError,
                   "ValueError": ValueError, "NameError": NameError,
                   "ArithmeticError": ArithmeticError, "AttributeError": AttributeError,
                   "RuntimeError": RuntimeError}.get(kind, RuntimeError)
        cls = type("Julia" + kind, (JuliaError, builtin), {"__module__": "scibmad"})
        _ERROR_CLASSES[kind] = cls
    return cls


def _convert_error(e, kind=None):
    if not isinstance(e, _JuliaCallError):
        return e
    try:
        k, jtype, msg = G.errorinfo(e.exception)
    except Exception:  # pragma: no cover
        k, jtype, msg = "RuntimeError", "Exception", str(e)
    if k == "KeyboardInterrupt":
        return KeyboardInterrupt()
    cls = _error_class(kind or k)
    err = cls(str(msg))
    err.julia_type = str(jtype)
    err.julia_exception = e.exception
    err.__cause__ = None
    return err


def _jcall(f, *args, **kwargs):
    """Calls a Julia function with Julia arguments, raising Python exceptions."""
    try:
        return f(*args, **kwargs)
    except _JuliaCallError as e:
        raise _convert_error(e) from None


# ---------------------------------------------------------------------------------------------
# Python -> Julia


class Symbol(str):
    """A Julia ``Symbol``: a string that is passed to Julia as a Symbol (``:name``)."""

    __slots__ = ()

    def __repr__(self):
        return f"Symbol({str.__repr__(self)})"


class _IndexSentinel:
    """``scibmad.end`` and ``scibmad.begin``: Julia's ``end`` and ``begin`` in indexing, with
    integer offsets (``scibmad.end - 1``)."""

    __slots__ = ("_kind", "_offset")

    def __init__(self, kind, offset=0):
        self._kind = kind
        self._offset = offset

    def __add__(self, k):
        if not isinstance(k, (int, np.integer)):
            return NotImplemented
        return _IndexSentinel(self._kind, self._offset + int(k))

    __radd__ = __add__

    def __sub__(self, k):
        if not isinstance(k, (int, np.integer)):
            return NotImplemented
        return _IndexSentinel(self._kind, self._offset - int(k))

    def __repr__(self):
        s = "scibmad." + self._kind
        if self._offset == 0:
            return s
        return f"{s} {'+' if self._offset > 0 else '-'} {abs(self._offset)}"

    def _to_julia(self):
        return (G.indexend if self._kind == "end" else G.indexbegin)(self._offset)


end = _IndexSentinel("end")
begin = _IndexSentinel("begin")

_PLAIN = (bool, int, float, complex, str)
_NUMERIC_KINDS = set("biufc")


def _is_number(x):
    return isinstance(x, (int, float, complex, np.number, np.bool_))


def to_julia(x):
    """The Julia value for a Python value (proxies give their Julia value)."""
    if x is None or type(x) in _PLAIN:
        return x
    if isinstance(x, JuliaObject):
        return object.__getattribute__(x, "_jl")
    if _ValueBase is not None and isinstance(x, _ValueBase):
        return x
    if isinstance(x, Symbol):
        return jl.Symbol(str(x))
    if isinstance(x, _PLAIN):  # subclasses (e.g. numpy.float64, bool subclasses, str subclasses)
        if isinstance(x, bool):
            return bool(x)
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return float(x)
        if isinstance(x, complex):
            return complex(x)
        return str(x)
    if isinstance(x, np.ndarray):
        return _array_to_julia(x)
    if isinstance(x, np.generic):
        return _scalar_to_julia(x)
    if isinstance(x, list):
        return _list_to_julia(x)
    if isinstance(x, tuple):
        return tuple(to_julia(v) for v in x)
    if isinstance(x, dict):
        if not x:
            return G.dict(G.vect(), G.vect())
        return G.dict(G.vect(*[to_julia(k) for k in x]), G.vect(*[to_julia(v) for v in x.values()]))
    if isinstance(x, (set, frozenset)):
        return G.set(G.vect(*[to_julia(v) for v in x]))
    if isinstance(x, range):
        return x  # juliacall gives a Julia range of the same integers
    if x is Ellipsis:
        return G.colon()
    if isinstance(x, slice):
        if x.start is None and x.stop is None and x.step is None:
            return G.colon()
        raise TypeError("A slice can only be used in indexing (or `slice(None)` / `...` for "
                        "Julia's `:`)")
    if isinstance(x, _IndexSentinel):
        return x._to_julia()
    if isinstance(x, re.Pattern):
        flags = ""
        if x.flags & re.IGNORECASE:
            flags += "i"
        if x.flags & re.MULTILINE:
            flags += "m"
        if x.flags & re.DOTALL:
            flags += "s"
        if x.flags & re.VERBOSE:
            flags += "x"
        return G.regex(x.pattern, flags)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, type):
        jt = _python_type_to_julia(x)
        return x if jt is None else jt
    if isinstance(x, np.dtype):
        jt = _python_type_to_julia(x.type)
        return x if jt is None else jt
    if callable(x):
        return G.PyFunction(_python_function(x))
    if hasattr(x, "__array__") or hasattr(x, "__array_interface__"):
        return _array_to_julia(np.asarray(x))
    return x


_TYPES = {"float": "Float64", "int": "Int64", "complex": "ComplexF64", "bool": "Bool",
          "str": "String", "float64": "Float64", "float32": "Float32", "float16": "Float16",
          "int64": "Int64", "int32": "Int32", "int16": "Int16", "int8": "Int8",
          "uint64": "UInt64", "uint32": "UInt32", "uint16": "UInt16", "uint8": "UInt8",
          "complex128": "ComplexF64", "complex64": "ComplexF32", "bool_": "Bool"}


def _python_type_to_julia(t):
    """The Julia type for a Python or NumPy number type (np.float32 -> Float32), or None."""
    if t in (float, int, complex, bool, str) or (isinstance(t, type) and
                                                  issubclass(t, (np.number, np.bool_))):
        name = _TYPES.get(t.__name__)
        if name is not None:
            return G.getname(jl.Base, jl.Symbol(name))
    return None


def _scalar_to_julia(x):
    # juliacall converts NumPy scalars to the Julia number of the same type (float32 -> Float32)
    if isinstance(x, (np.number, np.bool_)):
        return x
    return x.item()


def _array_to_julia(a):
    if a.ndim == 0:
        return to_julia(a[()])
    if a.dtype.kind in _NUMERIC_KINDS:
        if not a.dtype.isnative:
            a = a.astype(a.dtype.newbyteorder("="))
        return G.toarray(a)
    # other element types: each entry converted, column-major as Julia stores it
    items = [to_julia(v) for v in a.ravel(order="F")]
    v = G.vect(*items)
    return v if a.ndim == 1 else G.reshapeto(v, a.shape)


def _list_to_julia(x):
    if not x:
        return G.vect()
    if all(_is_number(v) for v in x):
        return _array_to_julia(np.asarray(x))
    if all(isinstance(v, list) for v in x) and all(v and all(_is_number(w) for w in v)
                                                   for v in x):
        a = np.asarray(x)
        if a.dtype != object:
            return _array_to_julia(a)
    return G.vect(*[to_julia(v) for v in x])


def _python_function(f):
    """A Python function that Julia can call: the arguments become proxies, the result a Julia
    value."""

    def adapter(*args, **kwargs):
        r = f(*[wrap(a) for a in args], **{k: wrap(v) for k, v in kwargs.items()})
        return to_julia(r)

    adapter.__name__ = getattr(f, "__name__", "function")
    adapter.__wrapped__ = f
    return adapter


def _index_to_julia(k):
    """An index of Python `obj[...]` for an object that is not an array (Julia's indexes;
    negative integers and `scibmad.end` count from the end)."""
    if isinstance(k, slice):
        if k.start is None and k.stop is None and k.step is None:
            return G.colon()
        bounds = (k.start, k.stop)
        if any(isinstance(b, (float, np.floating)) for b in bounds):
            if k.step is not None:
                raise TypeError("An s-range (a slice of floats) has no step")
            a = -np.inf if k.start is None else float(k.start)
            b = np.inf if k.stop is None else float(k.stop)
            return (a, b)
        step = 1 if k.step is None else int(k.step)
        if step == 0:
            raise ValueError("slice step cannot be zero")
        return G.pyslice(None if k.start is None else _index_to_julia(k.start),
                         None if k.stop is None else _index_to_julia(k.stop), step)
    if isinstance(k, list):
        return G.vect(*[_index_to_julia(v) for v in k]) if k else G.vect()
    if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
        return int(k)
    return to_julia(k)


def _kwargs_to_julia(kw):
    return {k: to_julia(v) for k, v in kw.items()}


# ---------------------------------------------------------------------------------------------
# Julia -> Python

_CACHE = weakref.WeakValueDictionary()  # identity of proxies of mutable Julia objects


def wrap(v):
    """The Python value for a value that juliacall returned."""
    t = type(v)
    if v is None or t in _PLAIN or t is range or t is Fraction:
        return v
    if t is tuple:
        return tuple(wrap(x) for x in v)
    if _ValueBase is not None and isinstance(v, _ValueBase):
        return _wrap_julia(v)
    return v


def _wrap_julia(v):
    cat, oid = G.wrapinfo(v)
    if cat == 6:  # Symbol
        return Symbol(str(v))
    if cat == 2:
        if G.isnumpyable(v):
            return np.asarray(v)
        if G.isnumeric(v) and not G.isdevice(v):
            return np.array(G.tohost(v))
    if oid:
        p = _CACHE.get(oid)
        if p is not None and G.egal(object.__getattribute__(p, "_jl"), v):
            return p
    cls = _CLASSES[cat]
    p = cls.__new__(cls)
    object.__setattr__(p, "_jl", v)
    if oid:
        _CACHE[oid] = p
    return p


def _display_size():
    opts = _OPTIONS
    if opts["display_width"] and opts["display_height"]:
        return opts["display_width"], opts["display_height"]
    if _in_notebook():
        w, h = 120, 40
    else:
        w, h = shutil.get_terminal_size((100, 30))
    return opts["display_width"] or w, opts["display_height"] or h


_OPTIONS = {"display_width": None, "display_height": None}


def _in_notebook():
    ip = sys.modules.get("IPython")
    if ip is None:
        return False
    try:
        shell = ip.get_ipython()
    except Exception:
        return False
    return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"


def _setup_ipython():
    if not _in_notebook():
        return
    shell = sys.modules["IPython"].get_ipython()
    try:
        G.redirect_to_python()
        shell.events.register("post_execute", lambda: G.flush_stdio())
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------------------------
# Proxies

_TRACING = [False]  # while a Python function is traced into a DefExpr (see _extras.py)


class _TracingAbort(Exception):
    pass


def _binop(name, a, b):
    return wrap(_jcall(_F[name], to_julia(a), to_julia(b)))


def _make_binops(cls, broadcast=False):
    """Operators as Julia's: elementwise (Julia broadcasting) for arrays and collections, as
    NumPy does, except `==` and `!=`, which compare whole values as in Julia."""
    ops = {"add": "+", "sub": "-", "mul": "*", "truediv": "/", "pow": "^", "mod": "mod",
           "floordiv": "fld", "and": "&", "or": "|", "xor": "xor"}
    for py, jlop in ops.items():
        def fwd(self, other, _op=jlop):
            if broadcast:
                return wrap(_jcall(_F["broadcast"], _F[_op], self._jl, to_julia(other)))
            return _binop(_op, self, other)

        def rev(self, other, _op=jlop):
            if broadcast:
                return wrap(_jcall(_F["broadcast"], _F[_op], to_julia(other), self._jl))
            return _binop(_op, other, self)

        setattr(cls, f"__{py}__", fwd)
        setattr(cls, f"__r{py}__", rev)
    cmps = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    for py, jlop in cmps.items():
        def cmp(self, other, _op=jlop):
            try:
                o = to_julia(other)
            except Exception:
                return NotImplemented
            if broadcast and _op not in ("==", "!="):
                return wrap(_jcall(_F["broadcast"], _F[_op], self._jl, o))
            return wrap(_jcall(_F[_op], self._jl, o))

        setattr(cls, f"__{py}__", cmp)
    if broadcast:
        for py, jlop in (("neg", "-"), ("pos", "+"), ("abs", "abs"), ("invert", "~")):
            def un(self, _op=jlop):
                return wrap(_jcall(_F["broadcast"], _F[_op], self._jl))

            setattr(cls, f"__{py}__", un)


class JuliaObject:
    """A Julia value. Attributes, indexing, calls and operators are those of Julia."""

    __slots__ = ("_jl", "__weakref__")

    def __init__(self, value):
        object.__setattr__(self, "_jl", to_julia(value))

    # attributes
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return wrap(G.getprop(object.__getattribute__(self, "_jl"), name))
        except _JuliaCallError as e:
            raise _convert_error(e, "AttributeError") from None

    def __setattr__(self, name, value):
        if name.startswith("_"):
            raise AttributeError(f"cannot set {name!r}")
        _jcall(G.setprop, self._jl, name, to_julia(value))

    def __dir__(self):
        try:
            props = list(G.pyproperties(self._jl))
        except Exception:
            props = []
        return sorted(set(props))

    # indexing and calls
    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return wrap(_jcall(G.index, self._jl, tuple(_index_to_julia(k) for k in key)))

    def __setitem__(self, key, value):
        key = key if isinstance(key, tuple) else (key,)
        _jcall(G.setindex, self._jl, to_julia(value), tuple(_index_to_julia(k) for k in key))

    def __call__(self, *args, **kwargs):
        if kwargs or args:
            jargs = tuple(_index_to_julia(a) if isinstance(a, (slice, list)) else to_julia(a)
                          for a in args)
        else:
            jargs = ()
        return wrap(_jcall(G.call, self._jl, jargs, **_kwargs_to_julia(kwargs)))

    # display
    def __repr__(self):
        w, h = _display_size()
        try:
            return str(G.repr_plain(self._jl, w, h, True))
        except Exception as e:  # pragma: no cover
            return f"<Julia value (show failed: {e})>"

    def __str__(self):
        return str(_jcall(G.str, self._jl))

    def _repr_html_(self):
        try:
            r = G.repr_html(self._jl)
        except Exception:
            return None
        return None if r is None else str(r)

    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))

    # protocols
    def __bool__(self):
        if _TRACING[0]:
            raise _TracingAbort
        try:
            return bool(G.truth(self._jl))
        except _JuliaCallError as e:
            raise _convert_error(e, "TypeError") from None

    def __hash__(self):
        return int(_jcall(_F["hash"], self._jl)) & 0x7FFFFFFFFFFFFFFF

    def __copy__(self):
        return wrap(_jcall(_F["copy"], self._jl))

    def __deepcopy__(self, memo):
        return wrap(_jcall(_F["deepcopy"], self._jl))

    def __neg__(self):
        return wrap(_jcall(_F["-"], self._jl))

    def __pos__(self):
        return wrap(_jcall(_F["+"], self._jl))

    def __abs__(self):
        return wrap(_jcall(_F["abs"], self._jl))

    def __invert__(self):
        return wrap(_jcall(_F["~"], self._jl))

    def __matmul__(self, other):
        return _binop("*", self, other)

    def __rmatmul__(self, other):
        return _binop("*", other, self)

    def __reduce__(self):
        raise TypeError(f"Julia values cannot be pickled ({type(self).__name__})")


_make_binops(JuliaObject)


# NumPy ufuncs as Julia functions (for number-like values: Taylor series, deferred
# expressions, ...)
_UFUNCS = {
    "add": "+", "subtract": "-", "multiply": "*", "true_divide": "/", "divide": "/",
    "power": "^", "float_power": "^", "negative": "-", "positive": "+", "absolute": "abs",
    "fabs": "abs", "sqrt": "sqrt", "cbrt": "cbrt", "exp": "exp", "exp2": "exp2", "expm1": "expm1",
    "log": "log", "log2": "log2", "log10": "log10", "log1p": "log1p", "sin": "sin", "cos": "cos",
    "tan": "tan", "arcsin": "asin", "arccos": "acos", "arctan": "atan", "arctan2": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "arcsinh": "asinh", "arccosh": "acosh",
    "arctanh": "atanh", "conjugate": "conj", "conj": "conj", "sign": "sign", "reciprocal": "inv",
    "hypot": "hypot", "maximum": "max", "minimum": "min", "fmax": "max", "fmin": "min",
    "floor": "floor", "ceil": "ceil", "trunc": "trunc", "rint": "round", "deg2rad": "deg2rad",
    "rad2deg": "rad2deg", "isnan": "isnan", "isinf": "isinf", "isfinite": "isfinite",
    "equal": "==", "not_equal": "!=", "less": "<", "less_equal": "<=", "greater": ">",
    "greater_equal": ">=",
}


def _julia_function(name):
    f = _F.get(name)
    if f is None:
        f = G.getname(jl.Base, jl.Symbol(name))
        _F[name] = f
    return f


class JuliaNumber(JuliaObject):
    """A number-like Julia value (Taylor series, deferred expressions, time-dependent and batch
    parameters, expressions of TwissExpr, ...): arithmetic, comparisons and NumPy functions
    (``np.sin(x)``) are those of Julia."""

    __slots__ = ()

    def __float__(self):
        if _TRACING[0]:
            raise _TracingAbort
        return float(_jcall(_F["Float64"], self._jl))

    def __complex__(self):
        if _TRACING[0]:
            raise _TracingAbort
        return complex(_jcall(_F["ComplexF64"], self._jl))

    def __int__(self):
        if _TRACING[0]:
            raise _TracingAbort
        return int(_jcall(_F["Int"], self._jl))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            return NotImplemented
        name = ufunc.__name__
        if name == "square":
            f = _F["*"]
            inputs = (inputs[0], inputs[0])
        else:
            jname = _UFUNCS.get(name)
            if jname is None:
                return NotImplemented
            f = _julia_function(jname)
        args = [to_julia(x) for x in inputs]
        if any(isinstance(x, np.ndarray) and x.ndim > 0 for x in inputs):
            return wrap(_jcall(_F["broadcast"], f, *args))
        return wrap(_jcall(f, *args))

    @property
    def real(self):
        return wrap(_jcall(_F["real"], self._jl))

    @property
    def imag(self):
        return wrap(_jcall(_F["imag"], self._jl))

    def conjugate(self):
        return wrap(_jcall(_F["conj"], self._jl))


class JuliaCollection(JuliaObject):
    """A Julia collection (a beamline, a table column, a dictionary, ...): ``len``, iteration and
    ``in`` are those of Julia, and ``numpy.asarray`` gives its values."""

    __slots__ = ()

    def __len__(self):
        return int(_jcall(_F["length"], self._jl))

    def __iter__(self):
        try:
            it = iter(self._jl)
        except _JuliaCallError as e:
            raise _convert_error(e) from None
        for v in it:
            yield wrap(v)

    def __contains__(self, item):
        return bool(_jcall(_F["in"], to_julia(item), self._jl))

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(wrap(_jcall(G.collectvalues, self._jl)))
        return a.astype(dtype) if dtype is not None else a

    def to_numpy(self):
        return self.__array__()


_make_binops(JuliaCollection, broadcast=True)


class JuliaArray(JuliaObject):
    """A Julia array that is not a NumPy array (its elements are not plain numbers, or it is on
    a GPU). It indexes like NumPy (from 0, slices, negative indexes), arithmetic is
    elementwise (Julia broadcasting, on the GPU for GPU arrays), and ``numpy.asarray`` copies it
    to the host."""

    __slots__ = ()

    @property
    def shape(self):
        return tuple(int(s) for s in G.arrsize(self._jl))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        s = self.shape
        return s[0] if s else 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item):
        return bool(_jcall(_F["in"], to_julia(item), self._jl))

    def _key(self, key):
        key = key if isinstance(key, tuple) else (key,)
        shape = self.shape
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            n = len(shape) - (len(key) - 1)
            key = key[:i] + (slice(None),) * n + key[i + 1:]
        if len(key) == 1 and len(shape) > 1 and not isinstance(key[0], slice):
            # a[i] of a matrix: the row (as NumPy)
            key = key + (slice(None),) * (len(shape) - 1)
        out = []
        for d, k in enumerate(key):
            n = shape[d] if d < len(shape) else 1
            out.append(_array_index(k, n))
        return tuple(out)

    def __getitem__(self, key):
        return wrap(_jcall(G.arrindex, self._jl, self._key(key)))

    def __setitem__(self, key, value):
        _jcall(G.arrsetindex, self._jl, to_julia(value), self._key(key))

    def __array__(self, dtype=None, copy=None):
        if G.isnumeric(self._jl):
            a = np.array(G.tohost(self._jl))
        else:
            flat = np.empty(self.size, dtype=object)
            for i, v in enumerate(G.vecvalues(self._jl)):  # Julia's (column-major) order
                flat[i] = wrap(v)
            a = flat.reshape(self.shape, order="F")
        return a.astype(dtype) if dtype is not None else a

    def to_numpy(self):
        return self.__array__()


_make_binops(JuliaArray, broadcast=True)


def _array_index(k, n):
    if isinstance(k, (bool, np.bool_)):
        raise IndexError("boolean scalar indexes are not supported")
    if isinstance(k, (int, np.integer)):
        i = int(k)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"index {k} is out of bounds for an axis of size {n}")
        return i + 1
    if isinstance(k, slice):
        start, stop, step = k.indices(n)
        r = range(start + 1, stop + 1, step)
        return r
    if isinstance(k, (list, np.ndarray)):
        a = np.asarray(k)
        if a.dtype == bool:
            return G.toarray(a)
        a = a.astype(np.int64)
        a = np.where(a < 0, a + n, a) + 1
        return G.toarray(a)
    if k is None:
        raise IndexError("numpy.newaxis is not supported for Julia arrays")
    raise IndexError(f"invalid index {k!r}")


class JuliaCallable(JuliaObject):
    """A Julia function or type. Calling it calls Julia (for a type: its constructor);
    ``isinstance(x, T)`` is Julia's ``x isa T``; ``T[P]`` is ``T{P}``."""

    __slots__ = ()

    def __get__(self, obj, objtype=None):
        # (makes `help()` show the Julia documentation: pydoc treats it as a routine)
        return self

    def __call__(self, *args, **kwargs):
        copyback = []
        jargs = []
        for a in args:
            j = to_julia(a)
            if isinstance(a, np.ndarray) and a.flags.writeable and a.ndim > 0:
                copyback.append((a, j))
            jargs.append(j)
        r = _jcall(G.callfunc, self._jl, tuple(jargs), **_kwargs_to_julia(kwargs))
        if copyback and self._mutating():
            # Julia functions named f! may change their arguments: copy the changes back into the
            # NumPy arrays that were copied for them
            for a, j in copyback:
                try:
                    np.copyto(a, np.asarray(wrap(j)), casting="unsafe")
                except Exception:
                    pass
        return wrap(r)

    def _mutating(self):
        try:
            return str(G.funcname(self._jl)).endswith("!")
        except Exception:
            return False

    def __getitem__(self, params):
        params = params if isinstance(params, tuple) else (params,)
        return wrap(_jcall(G.applytype, self._jl, tuple(to_julia(p) for p in params)))

    def __instancecheck__(self, obj):
        try:
            return bool(_F["isa"](to_julia(obj), self._jl))
        except Exception:
            return False

    @property
    def __doc__(self):
        try:
            return str(G.julia_doc(self._jl))
        except Exception:
            return None

    @property
    def __name__(self):
        try:
            return str(G.funcname(self._jl))
        except Exception:
            return "julia"


class JuliaModule(JuliaObject):
    """A Julia module. Its names are attributes; Julia's ``f!`` is ``f_`` in Python (or ``f!``
    through ``module["f!"]``, which gives any name, e.g. operators)."""

    __slots__ = ()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        s = G.jlname(self._jl, name)
        if s is None:
            raise AttributeError(f"Julia module {self} has no name {name!r}")
        return wrap(_jcall(G.getname, self._jl, s))

    def __getitem__(self, name):
        return wrap(_jcall(G.getname, self._jl, jl.Symbol(name)))

    def __dir__(self):
        names = set(str(n) for n in G.pynames(self._jl))
        names |= set(str(n) for n in G.modulenames(self._jl))
        return sorted(n for n in names if n.isidentifier())

    def __repr__(self):
        return f"<Julia module {self}>"


_CLASSES = {0: JuliaObject, 1: JuliaNumber, 2: JuliaArray, 3: JuliaCollection,
            4: JuliaCallable, 5: JuliaModule}
