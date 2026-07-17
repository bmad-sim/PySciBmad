"""Julia runtime bootstrap for the ``scibmad`` package.

Importing this module starts (or attaches to) the Julia session managed by
``juliacall``, loads ``SciBmad`` and ``PythonCall``, and includes the Julia glue
in :mod:`scibmad._glue`.

The heavy lifting -- installing the LTS Julia and the registered ``SciBmad.jl``
-- is done by ``juliapkg`` from :mod:`scibmad`'s ``juliapkg.json`` the first time
``juliacall`` is imported.
"""

from __future__ import annotations

import os

# Importing juliacall triggers juliapkg resolution (Julia + SciBmad install).
from juliacall import Main as jl  # noqa: E402

__all__ = [
    "jl",
    "SciBmad",
    "glue",
    "OP",
    "UNARY_OP",
]

# Make the packages available in the session. ``import`` (rather than ``using``)
# keeps Main uncluttered while still exposing the module objects we dispatch on.
jl.seval("import SciBmad")
jl.seval("import PythonCall")

#: The ``SciBmad`` Julia module. All exported SciBmad names hang off this and are
#: surfaced to Python via ``scibmad.__getattr__``.
SciBmad = jl.SciBmad

#: Subpackage modules, used as fall-backs when a reexported name is ambiguous at
#: the ``SciBmad`` level (e.g. ``Context``, which several subpackages export with
#: different meanings -- Beamlines' is the one accelerator users want).
Beamlines = jl.SciBmad.Beamlines
BeamTracking = jl.SciBmad.BeamTracking
AtomicAndPhysicalConstants = jl.SciBmad.AtomicAndPhysicalConstants

#: Lookup order for surfacing names to Python (most specific accelerator-facing
#: modules first).
LOOKUP_MODULES = (SciBmad, Beamlines, BeamTracking, AtomicAndPhysicalConstants)

# Load the Julia glue into its own module so it does not collide with anything.
_glue_path = os.path.join(os.path.dirname(__file__), "_glue.jl")
jl.seval("module _PySciBmadGlue end")
glue = jl._PySciBmadGlue
jl.Base.include(glue, _glue_path)

# Julia operator functions, fetched once, used by the Python operand wrappers to
# dispatch arithmetic onto the underlying Julia values.
OP = {
    "+": jl.seval("+"),
    "-": jl.seval("-"),
    "*": jl.seval("*"),
    "/": jl.seval("/"),
    "^": jl.seval("^"),
    "%": jl.seval("%"),
}

# Unary math functions exposed at the package top level (``scibmad.sin`` etc.).
# These come straight from Base and dispatch correctly onto the deferred types,
# which define them.
_UNARY_NAMES = (
    "sqrt", "exp", "log", "log10", "sin", "cos", "tan", "cot",
    "sinh", "cosh", "tanh", "coth", "asin", "acos", "atan", "acot",
    "asinh", "acosh", "atanh", "acoth", "sinc", "csc", "csch", "sec",
    "sech", "abs", "sign", "conj", "inv", "zero", "one",
)
UNARY_OP = {name: jl.seval(name) for name in _UNARY_NAMES}
