# Julia side of the Python interface. It is included into a module of its own (`PySciBmad`) in
# Main when Python starts Julia. It knows nothing about SciBmad's API beyond the few functions
# whose Python form must differ from the Julia one (`Context` stacks for `withvalues`, `DefExpr`
# tracing): everything else is reached generically by name, so that new functionality of the
# Julia packages is available in Python without changes here.
module PySciBmad

using PythonCall
using SciBmad

# ---------------------------------------------------------------------------------------------
# Names

# The Julia name of a Python attribute name of module `m`: the name itself if `m` has it, else
# `name!` for `name_` (`match_` -> `match!`), else nothing.
function jlname(m::Module, name::AbstractString)
  s = Symbol(name)
  isdefined(m, s) && return s
  if endswith(name, "_")
    b = Symbol(chop(name) * "!")
    isdefined(m, b) && return b
  end
  return nothing
end

# The Python names of the exported names of `m`: `f!` is `f_`; names that are not Python
# identifiers (operators, macros) are left out.
function pynames(m::Module)
  out = String[]
  for s in names(m)
    n = String(s)
    endswith(n, "!") && (n = chop(n) * "_")
    Base.isidentifier(n) && isascii(n) && push!(out, n)
  end
  return out
end

getname(m::Module, s::Symbol) = getproperty(m, s)

# ---------------------------------------------------------------------------------------------
# Categories of values, which decide the Python class of their proxy (cached per type)
#   1: number-like (arithmetic with numbers: TPS, DefExpr, TimeDependentParam, BatchParam,
#      LatticeExpr, ...), 2: array, 3: collection (length and iteration), 4: function or type,
#   5: module, 6: Symbol, 0: anything else
const CATEGORY = IdDict{Any,Int}()

category(x) = get!(() -> _category(typeof(x)), CATEGORY, typeof(x))
function _category(T)
  T <: Module && return 5
  (T <: Function || T <: Type) && return 4
  T <: AbstractArray && return 2
  T <: Symbol && return 6
  T <: AbstractString && return 0
  (T <: Number || hasmethod(+, Tuple{T,Float64})) && return 1
  (hasmethod(length, Tuple{T}) && hasmethod(iterate, Tuple{T})) && return 3
  return 0
end

# The category, and the objectid of a value that is not a bits value (0 otherwise), so that
# the same Julia object gives the same Python proxy (`is` in Python is `===` in Julia)
wrapinfo(x) = (category(x),
               (isbits(x) && !(x isa Function)) || isnumpyable(x) ? UInt(0) : objectid(x))
egal(a, b) = a === b

const PlainNumber = Union{Bool,Base.BitInteger,Base.IEEEFloat,Complex{<:Base.IEEEFloat}}
# Whether an array can be shared with NumPy without a copy (a CPU array of plain numbers)
isnumpyable(x) = false
isnumpyable(x::Array{<:PlainNumber}) = true
# Whether an array holds plain numbers (it is then copied to a NumPy array of numbers)
isnumeric(x::AbstractArray) = eltype(x) <: PlainNumber
isnumeric(x) = false
# A CPU copy of an array (GPU arrays are copied to the host)
tohost(x::AbstractArray) = Array(x)
# Whether an array is on a GPU (KernelAbstractions knows its backend and it is not the CPU)
const KA = Base.PkgId(Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c"), "KernelAbstractions")
function isdevice(x::AbstractArray)
  x isa Array && return false
  mod = get(Base.loaded_modules, KA, nothing)
  isnothing(mod) && return false
  b = try
    mod.get_backend(x)
  catch
    return false
  end
  return !(b isa mod.CPU)
end
collectvalues(x) = collect(x)
vecvalues(x::AbstractArray) = vec(Array(x))

# ---------------------------------------------------------------------------------------------
# Building Julia values from Python ones

# A vector as the Julia literal [xs...] gives it (the element type is the promoted one)
vect(xs...) = [xs...]
dict(ks::AbstractVector, vs::AbstractVector) =
  isempty(ks) ? Dict{Any,Any}() : Dict(k => v for (k, v) in zip(ks, vs))
set(xs::AbstractVector) = Set(xs)
regex(p::AbstractString, flags::AbstractString) = Regex(p, flags)
colon() = Colon()
# An array of the element type that the Julia literal of its entries would have
narrow(a::AbstractArray) = isempty(a) ? a : identity.(a)
reshapeto(v::AbstractVector, dims) = reshape(v, Tuple(dims))
# Copies a NumPy array (seen by Julia as a PyArray) into an Array of its element type
toarray(a::AbstractArray{T}) where {T} = Array{T}(a)
function toarray(a::AbstractArray{Py})
  b = Array{Any}(undef, size(a))
  for i in eachindex(a, b)
    b[i] = pyconvert(Any, a[i])
  end
  return narrow(b)
end

# A Python function as a Julia function. The Python side wraps the arguments into proxies and
# converts the result back to a Julia value, which is taken out of its Python wrapper here.
struct PyFunction <: Function
  f::Py
end
(p::PyFunction)(args...; kwargs...) = pyconvert(Any, p.f(args...; kwargs...))
Base.show(io::IO, p::PyFunction) = print(io, "PyFunction(", p.f, ")")

# ---------------------------------------------------------------------------------------------
# Indexing

# Python `obj[...]` for objects that are not arrays: the indexes are those of Julia, except that
# negative integers count from the end (-1 is `end`), `End(k)`/`Begin(k)` are `end + k`/
# `begin + k`, and slices of integers are Python's (the stop excluded).
struct IndexEnd
  offset::Int
end
struct IndexBegin
  offset::Int
end
indexend(k) = IndexEnd(k)
indexbegin(k) = IndexBegin(k)
_lastindex(x, d, n) =
  n == 1 ? lastindex(x) : applicable(lastindex, x, d) ? lastindex(x, d) : lastindex(x)
_firstindex(x, d, n) =
  n == 1 ? firstindex(x) : applicable(firstindex, x, d) ? firstindex(x, d) : firstindex(x)
resolve(x, d, n, i) = i
resolve(x, d, n, i::Integer) = i < 0 ? _lastindex(x, d, n) + 1 + i : i
resolve(x, d, n, i::IndexEnd) = _lastindex(x, d, n) + i.offset
resolve(x, d, n, i::IndexBegin) = _firstindex(x, d, n) + i.offset
# (an element index with a position: the index itself is resolved)
resolve(x, d, n, v::AbstractVector) = [resolve(x, d, n, i) for i in v]
struct PySlice
  start::Any
  stop::Any
  step::Int
end
pyslice(a, b, s) = PySlice(a, b, s)
function resolve(x, d, n, s::PySlice)
  a = isnothing(s.start) ? (s.step > 0 ? _firstindex(x, d, n) : _lastindex(x, d, n)) :
      resolve(x, d, n, s.start)
  b = isnothing(s.stop) ? (s.step > 0 ? _lastindex(x, d, n) : _firstindex(x, d, n)) :
      resolve(x, d, n, s.stop) - sign(s.step)
  return s.step == 1 ? (a:b) : (a:s.step:b)
end

function index(x, args::Tuple; kwargs...)
  n = length(args)
  return getindex(x, ntuple(d -> resolve(x, d, n, args[d]), n)...; kwargs...)
end
function setindex(x, v, args::Tuple)
  n = length(args)
  setindex!(x, v, ntuple(d -> resolve(x, d, n, args[d]), n)...)
  return nothing
end

# Python `obj(...)`: a call if `obj` can be called (a function, a type, or a callable object such
# as a `DefExpr`), else indexing with keywords, which Python has no syntax for:
# `res.x(qd, turn=2)` is `res.x[qd, turn=2]`.
const CALLABLE = IdDict{Any,Bool}()
iscallable(x) = x isa Union{Function,Type} ||
                get!(() -> !isempty(methods(x)), CALLABLE, typeof(x))
call(x, args::Tuple; kwargs...) = iscallable(x) ? x(args...; kwargs...) : index(x, args; kwargs...)
funcname(f) = string(nameof(f))

# A call of a function from Python. Python strings are Julia Strings; where the function has no
# method for a String but one for a Symbol, they are Symbols (Python has no Symbols:
# `rawget(ele, "Kn1")` is `rawget(ele, :Kn1)`, `match!(...; solver="levenberg_marquardt")` is
# `solver=:levenberg_marquardt`).
function callfunc(f, args::Tuple; kwargs...)
  kw = values(kwargs)
  for _ in 0:length(kw)
    try
      return f(args...; kw...)
    catch e
      if e isa TypeError && e.func === Symbol("keyword argument") && e.got isa AbstractString &&
         e.context isa Symbol && haskey(kw, e.context) && Symbol <: e.expected
        kw = merge(kw, NamedTuple{(e.context,)}((Symbol(e.got),)))
        continue
      end
      if e isa MethodError && e.f === f && any(a -> a isa AbstractString, args)
        sargs = map(a -> a isa AbstractString ? Symbol(a) : a, args)
        hasmethod(f, typeof(sargs)) && return f(sargs...; kw...)
      end
      rethrow()
    end
  end
end
applytype(T, params::Tuple) = T{params...}

# NumPy-style indexing of Julia arrays (the Python side makes the indexes 1-based)
arrindex(a, idx::Tuple) = a[idx...]
function arrsetindex(a, v, idx::Tuple)
  all(i -> i isa Integer, idx) ? (a[idx...] = v) : (a[idx...] .= v)
  return nothing
end
arrsize(a) = size(a)

# ---------------------------------------------------------------------------------------------
# Python protocols

function truth(x)
  x isa Bool && return x
  x isa Number && return !iszero(x)
  x isa SciBmad.Target &&
    error("The truth value of a Target is not defined. A chained comparison lo <= x <= hi " *
          "is two targets in Python as in Julia: write Target(x, lower=lo, upper=hi)")
  category(x) == 1 &&
    error("The truth value of a $(nameof(typeof(x))) is not defined")
  applicable(isempty, x) && return !isempty(x)
  return true
end

function repr_plain(x, width::Integer, height::Integer, limit::Bool)
  io = IOBuffer()
  ctx = IOContext(io, :limit => limit, :displaysize => (Int(height), Int(width)),
                  :color => false, :python => true)
  show(ctx, MIME"text/plain"(), x)
  return String(take!(io))
end
repr_html(x) = showable(MIME"text/html"(), x) ? sprint(show, MIME"text/html"(), x) : nothing
str(x) = sprint(print, x; context=:python => true)

pyproperties(x) = String[String(s) for s in propertynames(x)]
getprop(x, name::AbstractString) = getproperty(x, Symbol(name))
setprop(x, name::AbstractString, v) = (setproperty!(x, Symbol(name), v); nothing)

julia_doc(x) = string(Base.Docs.doc(x))
julia_doc(m::Module, s::Symbol) = string(Base.Docs.doc(Base.Docs.Binding(m, s)))

# The Python exception class name and the message of a Julia exception
function errorinfo(e)
  msg = try
    sprint(showerror, e)
  catch
    string(e)
  end
  kind = e isa BoundsError ? "IndexError" :
         e isa KeyError ? "KeyError" :
         e isa MethodError ? "TypeError" :
         e isa Union{ArgumentError,DomainError,InexactError,DimensionMismatch} ? "ValueError" :
         e isa UndefVarError ? "NameError" :
         e isa Union{OverflowError,DivideError} ? "ArithmeticError" :
         e isa InterruptException ? "KeyboardInterrupt" : "RuntimeError"
  return (kind, String(nameof(typeof(e))), msg)
end

# ---------------------------------------------------------------------------------------------
# The few functions whose Python form differs

# Context stacks for the Python `with withvalues(ctx, ...)` block, as Beamlines' `withvalues`
# restores them (also what the block pushed and did not pop)
nsaved(c) = length(getfield(c, :saved))
function popto(c, n::Integer)
  while length(getfield(c, :saved)) > n
    pop!(c)
  end
  return nothing
end

# Deferred expressions traced from Python functions: the function is called once with a
# context whose variables are these expressions, and the operations on them build the
# `DefExpr` in Julia, so that no Python is called when it is evaluated.
function ctxvar(name::AbstractString)
  s = Symbol(name)
  return DefExpr((c=SciBmad.Beamlines.NULL_CONTEXT) -> getproperty(c, s))
end
ctxindex(d, i::Integer) = DefExpr((c=SciBmad.Beamlines.NULL_CONTEXT) -> d(c)[i])
ctxconst(x) = DefExpr((c=SciBmad.Beamlines.NULL_CONTEXT) -> x)
isdefexpr(x) = x isa DefExpr
iselement(x) = x isa SciBmad.LatElement

# ---------------------------------------------------------------------------------------------
# Julia's stdout and stderr to Python's (in Jupyter, where they are not the process's), as
# juliacall's IPython extension does it; flushed after each cell.
const REDIRECT = Ref{Any}(nothing)
function redirect_to_python()
  isnothing(REDIRECT[]) || return nothing
  out = redirect_stdout()
  err = redirect_stderr()
  pyout = PyIO(pyimport("sys" => "stdout"); line_buffering=true)
  pyerr = PyIO(pyimport("sys" => "stderr"); line_buffering=true)
  t1 = @async write(pyout, out)
  t2 = @async write(pyerr, err)
  REDIRECT[] = (out, err, pyout, pyerr, t1, t2)
  return nothing
end
function flush_stdio()
  r = REDIRECT[]
  flush(stdout)
  flush(stderr)
  if !isnothing(r)
    yield()
    flush(r[3])
    flush(r[4])
  end
  return nothing
end

# ---------------------------------------------------------------------------------------------
# Packages

const PKG = Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg")

isinstalled(name::AbstractString) = !isnothing(Base.find_package(name))
function install(name::AbstractString)
  Pkg = Base.require(PKG)
  Base.invokelatest(Pkg.add, name)
  return nothing
end
pkguuid(name::AbstractString) = string(Base.identify_package(name).uuid)
load(name::AbstractString) = Base.invokelatest(Base.require, Main, Symbol(name))

# A Julia file (e.g. a lattice) run in a module of its own, which `using SciBmad` (its variables
# are then the module's names)
function include_file(path::AbstractString)
  m = Module(Symbol("lattice_", splitext(basename(path))[1]))
  Base.eval(m, :(using SciBmad))
  Base.invokelatest(Base.include, m, abspath(path))
  return m
end
# All the names a module defines (not only its exported ones)
modulenames(m::Module) = String[String(s) for s in names(m; all=true)
                                if !startswith(String(s), "#") && s !== nameof(m) &&
                                   !(s in (:eval, :include))]

version(m::Module) = string(pkgversion(m))

end # module
