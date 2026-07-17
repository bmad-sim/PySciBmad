# Julia-side glue for the `scibmad` Python package.
#
# This file is `include`d into an anonymous module by _runtime.py after
# `using SciBmad` and `using PythonCall`. It provides the small amount of Julia
# code needed to build SciBmad's deferred-parameter types from *Python*
# callables/values, which the generic Julia constructors cannot do on their own
# (they inspect `applicable`/argument types, which behave differently for the
# `Py` objects that juliacall hands them).
#
# Everything here is intentionally minimal. The long-term home for this logic is
# a `PythonCall` package extension in the relevant SciBmad subpackage (as is
# already done for `Beamlines` in `BeamlinesPythonCallExt`); keeping it here lets
# the Python package work today against the *registered* Julia packages without
# requiring changes to them.

using SciBmad
using PythonCall

const Beamlines = SciBmad.Beamlines
const BeamTracking = SciBmad.BeamTracking

const DefExpr = Beamlines.DefExpr
const Context = Beamlines.Context
const TimeDependentParam = BeamTracking.TimeDependentParam
const BatchParam = BeamTracking.BatchParam

# --- DefExpr -----------------------------------------------------------------
#
# A DefExpr wraps a lambda that is either 0-argument or accepts a `Context`.
# We build the two forms explicitly so that the returned DefExpr closes over the
# *real* Context at evaluation time (a Python lambda passed straight to
# `DefExpr(f)` would always be treated as 0-argument because juliacall's `Py`
# is callable with any number of arguments).

# 0-argument Python callable, e.g. `DefExpr(lambda: 1)`.
function defexpr_from_py0(pyf, ::Type{T}) where {T}
  return DefExpr{T}(() -> pyconvert(T, pyf()))
end

# 1-argument (Context-accepting) Python callable, e.g. `DefExpr(lambda c: c.k1)`.
# The strict `::Context` annotation forces Beamlines' constructor down its
# context-aware branch so the real Context reaches Python at evaluation time.
function defexpr_from_py1(pyf, ::Type{T}) where {T}
  return DefExpr{T}((c::Context) -> pyconvert(T, pyf(c)))
end

# --- TimeDependentParam ------------------------------------------------------
#
# `TimeDependentParam(f::Function, isconst)` expects a Julia function. Wrap the
# Python callable so it is called as `f(t)` and its result converted to T.
function tdparam_from_py(pyf, ::Type{T}) where {T}
  return TimeDependentParam(t -> pyconvert(T, pyf(t)), false)
end

# --- BatchParam --------------------------------------------------------------
#
# `BatchParam(::AbstractArray)` needs a genuine Julia array; a Python list
# arrives as a `Py`. Convert element-by-element into a Vector{T}.
function batchparam_from_py(pyseq, ::Type{T}) where {T}
  v = pyconvert(Vector{T}, pyseq)
  return BatchParam(v)
end

# Scalar (degenerate) batch, from a Python number.
batchparam_scalar(v, ::Type{T}) where {T} = BatchParam(pyconvert(T, v))

# --- helpers -----------------------------------------------------------------

# Convert a Python sequence into a Julia `Vector{Any}` (used e.g. for the
# `twiss(..., at=[...])` argument, which dispatches on `::Vector`).
to_any_vector(pyseq) = collect(Any, pyseq)

# Type queries used by the Python wrapper to decide whether a returned Julia
# value should be re-wrapped as an operator-overloading operand.
is_defexpr(x) = x isa DefExpr
is_tdparam(x) = x isa TimeDependentParam
is_batchparam(x) = x isa BatchParam
is_deferred(x) = is_defexpr(x) || is_tdparam(x) || is_batchparam(x)
