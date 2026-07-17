"""Python operator-overloading wrappers for SciBmad's deferred-parameter types.

``juliacall`` only wires Python's arithmetic dunder methods to Julia values whose
type is a subtype of ``Number``. SciBmad's ``DefExpr``, ``TimeDependentParam`` and
``BatchParam`` define ``+ - * / ^`` (and unary math) in Julia but are *not*
``Number`` subtypes, so straight from ``juliacall`` this fails::

    sb.DefExpr(lambda: 1) + 10          # TypeError without this wrapper

:class:`Operand` fixes that by holding the underlying Julia value and forwarding
each Python operator to the corresponding Julia operator. Results that are still
deferred types are re-wrapped; results that are plain numbers pass straight
through.
"""

from __future__ import annotations

from ._runtime import OP, UNARY_OP, glue, jl

__all__ = ["Operand", "unwrap", "wrap"]


def unwrap(x):
    """Return the underlying Julia value for an :class:`Operand`, recursing into
    lists/tuples; everything else is returned unchanged for juliacall to handle."""
    if isinstance(x, Operand):
        return x._jl
    if isinstance(x, (list, tuple)):
        return [unwrap(i) for i in x]
    return x


def wrap(jlval):
    """Wrap a Julia value in an :class:`Operand` iff it is a deferred type.

    Plain numbers (and anything else juliacall already converts nicely) are
    returned as-is.
    """
    try:
        deferred = bool(glue.is_deferred(jlval))
    except Exception:
        deferred = False
    return Operand(jlval) if deferred else jlval


class Operand:
    """A Python handle to a Julia ``DefExpr`` / ``TimeDependentParam`` /
    ``BatchParam`` that supports Python arithmetic and (where applicable) calling.
    """

    __slots__ = ("_jl",)

    def __init__(self, jlval):
        self._jl = jlval

    # -- introspection --------------------------------------------------------
    @property
    def julia(self):
        """The underlying (juliacall-wrapped) Julia value."""
        return self._jl

    def __repr__(self):
        return f"<scibmad.Operand {self._jl!r}>"

    def __str__(self):
        return str(self._jl)

    # Forward attribute access (e.g. reading struct fields) to the Julia value.
    def __getattr__(self, name):
        # __slots__ means normal attributes never reach here except real misses.
        if name == "_jl":
            raise AttributeError(name)
        return getattr(self._jl, name)

    # -- calling (DefExpr()/DefExpr(ctx)/TimeDependentParam(t)) ---------------
    def __call__(self, *args):
        return wrap(self._jl(*[unwrap(a) for a in args]))

    # -- binary operators -----------------------------------------------------
    def _binop(self, other, sym):
        return wrap(OP[sym](self._jl, unwrap(other)))

    def _rbinop(self, other, sym):
        return wrap(OP[sym](unwrap(other), self._jl))

    def __add__(self, other):
        return self._binop(other, "+")

    def __radd__(self, other):
        return self._rbinop(other, "+")

    def __sub__(self, other):
        return self._binop(other, "-")

    def __rsub__(self, other):
        return self._rbinop(other, "-")

    def __mul__(self, other):
        return self._binop(other, "*")

    def __rmul__(self, other):
        return self._rbinop(other, "*")

    def __truediv__(self, other):
        return self._binop(other, "/")

    def __rtruediv__(self, other):
        return self._rbinop(other, "/")

    def __pow__(self, other):
        return self._binop(other, "^")

    def __rpow__(self, other):
        return self._rbinop(other, "^")

    def __mod__(self, other):
        return self._binop(other, "%")

    def __rmod__(self, other):
        return self._rbinop(other, "%")

    # -- unary operators ------------------------------------------------------
    def __neg__(self):
        return wrap(OP["-"](self._jl))

    def __pos__(self):
        return wrap(OP["+"](self._jl))

    def __abs__(self):
        return wrap(UNARY_OP["abs"](self._jl))
