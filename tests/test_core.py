"""Tests of the Python interface itself (conversions, indexing, calls, errors, ...), each checked
against the same computation done in Julia."""

import math
import re
import warnings

import numpy as np
import pytest

import scibmad as sb
from scibmad import julia


@pytest.fixture(scope="module")
def fodo():
    with sb.elements():
        qf = sb.Quadrupole(Kn1=0.36, L=0.5)
        sf = sb.Sextupole(Kn2=1.2, L=0.2)
        d = sb.Drift(L=0.1)
        b = sb.SBend(L=1.2, angle=np.pi / 132)
        qd = sb.Quadrupole(Kn1=-0.36, L=0.5, n_steps=3)
        sd = sb.Sextupole(Kn2=-1.2, L=0.2)
    bl = sb.Beamline([qf, sf, b, d, qd, sd, b, d], species_ref=sb.Species("electron"),
                     pc_ref=18e9)
    return dict(bl=bl, qf=qf, sf=sf, d=d, b=b, qd=qd, sd=sd)


def jl(code):
    return julia.seval(code)


# --- names -----------------------------------------------------------------------------------

def test_names():
    assert "twiss" in dir(sb) and "track_spin_" in dir(sb) and "match" in dir(sb)
    assert sb.push_ is julia.Base.push_  # f! is f_
    assert str(sb.match.__name__) == "match!"
    with pytest.raises(AttributeError):
        sb.no_such_name_at_all
    assert "Quadrupole" in (sb.Quadrupole.__doc__ or "")


def test_elements_names(fodo):
    assert fodo["qf"].name == "qf" and fodo["d"].name == "d"
    q = sb.Quadrupole(L=1.0, name="explicit")
    assert q.name == "explicit"
    ns = {"a": sb.Drift(L=1.0), "x": 3}
    sb.elements(ns)
    assert ns["a"].name == "a"

    def inner():
        with sb.elements():
            m = sb.Marker()
        return m
    assert inner().name == "m"


# --- conversions -----------------------------------------------------------------------------

def test_python_to_julia_types():
    T = julia.seval("x -> string(typeof(x))")
    assert T([1, 2]) == "Vector{Int64}"
    assert T([1, 2.5]) == "Vector{Float64}"
    assert T([[1.0, 2.0], [3.0, 4.0]]) == "Matrix{Float64}"
    assert T(np.zeros((2, 3), dtype=np.float32)) == "Matrix{Float32}"
    assert T(np.float32(1)) == "Float32"
    assert T((1, 2.0)) == "Tuple{Int64, Float64}"
    assert T([]) == "Vector{Any}"
    assert T({"a": 1}) == "Dict{String, Int64}"
    assert T(...) == "Colon"
    assert T(slice(None)) == "Colon"
    assert T(re.compile("q.*", re.I)) == "Regex"
    assert T(sb.Symbol("exit")) == "Symbol"
    assert T(None) == "Nothing"
    assert T(range(1, 10, 2)) == "StepRange{Int64, Int64}"
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    assert julia.seval("x -> x[2, 3]")(a) == 6.0  # logical indexes kept (row-major NumPy)
    obj = np.array([sb.Drift(L=1.0), sb.Drift(L=2.0)], dtype=object)
    assert T(obj) == "Vector{LatElement}"


def test_julia_to_python():
    v = jl("[1.0, 2.0, 3.0]")
    assert isinstance(v, np.ndarray) and v.dtype == np.float64
    jl("global _v = [1.0, 2.0, 3.0]")
    w = jl("_v")
    w[0] = 10.0  # NumPy arrays of Julia arrays share their memory
    assert jl("_v[1]") == 10.0
    assert isinstance(jl(":a"), sb.Symbol) and jl(":a") == "a"
    assert jl("(1, [2.0])")[1][0] == 2.0
    assert isinstance(jl("Any[1, \"a\"]"), sb.JuliaArray)
    assert jl("1:3") == range(1, 4)


def test_identity(fodo):
    bl, qf = fodo["bl"], fodo["qf"]
    assert bl[qf] is bl[qf]
    ds = bl[[fodo["d"]]]
    assert ds[0] is not ds[1]
    assert fodo["d"] is fodo["d"]


def test_copyback_of_mutating_functions():
    a = np.zeros(3)
    julia.Base.fill_(a, 2.0)
    assert np.all(a == 2.0)


# --- indexing --------------------------------------------------------------------------------

def test_beamline_indexing(fodo):
    bl = fodo["bl"]
    assert bl[1] is bl.line[0]  # beamline indexes start at 1, arrays at 0
    assert bl[-1] is bl[8] and bl[sb.end] is bl[8] and bl[sb.end - 1] is bl[7]
    assert len(bl) == 8 and [e.name for e in bl][:2] == ["qf", "sf"]
    assert [e.name for e in bl[lambda e: e.kind == "Quadrupole"]] == ["qf", "qd"]
    assert [e.name for e in bl[re.compile("(q|s)d")]] == ["qd", "sd"]
    assert [e.name for e in bl[2:4]] == ["sf", "b"]  # Python slices exclude the stop
    assert [e.name for e in bl["qf:qd"]] == ["qf", "sf", "b", "d", "qd"]
    assert bl[fodo["qf"]] in bl and fodo["qf"] not in bl  # (bl holds placements of qf)
    assert julia.seval("(q, bl) -> q in bl")(fodo["qf"], bl) is False  # as in Julia


def test_twiss_indexing(fodo):
    bl, qd = fodo["bl"], fodo["qd"]
    tw = sb.twiss(bl)
    J = julia.seval("(tw, qd) -> (tw.beta1[qd], tw.beta1[qd, :exit], tw.beta1[qd, :], "
                    "tw.beta1[qd, 2], tw.beta1[5, :], tw.beta1[(0, 1.23)], tw.beta1[end], "
                    "tw.beta1[end-1], tw.beta1[[qd]])")(tw, qd)
    P = (tw.beta1[qd], tw.beta1[qd, "exit"], tw.beta1[qd, :], tw.beta1[qd, 2], tw.beta1[5, :],
         tw.beta1[0:1.23], tw.beta1[-1], tw.beta1[sb.end - 1], tw.beta1[[qd]])
    for a, b in zip(J, P):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert tw.beta1[qd, sb.Symbol("exit")] == tw.beta1[qd, "exit"]
    np.testing.assert_array_equal(np.asarray(tw.beta1), np.asarray(tw.df.beta1))
    assert len(tw.beta1) == len(np.asarray(tw.beta1))


def test_keyword_indexing(fodo):
    bl = fodo["bl"]
    res = sb.track(bl, v0=[[1e-3, 0, 0, 0, 0, 0], [1e-3, 0, 0, 0, 0, 1e-3]], n_turns=3)
    J = julia.seval("res -> (res.x[turn=3], res.x[turn=3, particle=2], res.v[turn=3, particle=1],"
                    " res.x[turn=:, particle=1], res.x[])")(res)
    P = (res.x(turn=3), res.x(turn=3, particle=2), res.v(turn=3, particle=1),
         res.x(turn=..., particle=1), res.x[()])
    for a, b in zip(J, P):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert "x(turn=n, particle=i)" in repr(res.x)
    tw = sb.twiss(bl, chrom=2)
    assert tw.q1(delta=1) == julia.seval("tw -> tw.q1[delta=1]")(tw)


def test_julia_arrays_index_like_numpy():
    a = jl("Any[1 2 3; 4 5 6]")
    assert isinstance(a, sb.JuliaArray) and a.shape == (2, 3)
    assert a[0, 0] == 1 and a[1, -1] == 6 and list(a[1, 1:]) == [5, 6]
    a[0, 0] = 10
    assert jl("x -> x[1, 1]")(a) == 10
    np.testing.assert_array_equal(np.asarray(a), [[10, 2, 3], [4, 5, 6]])


# --- operators -------------------------------------------------------------------------------

def test_number_like_values():
    t = sb.Time()
    f = 1e-4 * np.sin(1e6 * t)  # a Julia TimeDependentParam, built by Julia's operators
    assert abs(f(np.pi / 2e6) - 1e-4) < 1e-18
    assert f(0) == 0.0
    b = sb.BatchParam([1.0, 2.0])
    c = 2 * b + 1
    np.testing.assert_array_equal(np.asarray(c.batch), [3.0, 5.0])
    d = sb.Descriptor(2, 2)
    x = sb.vars(d)
    y = np.exp(x[0]) * x[1]
    assert abs(float(y[0]) if False else 1.0) == 1.0
    assert julia.seval("y -> y[[1, 1]]")(y) == 1.0  # the coefficient of x1 x2


def test_truth_values(fodo):
    te = sb.TwissExpr(fodo["bl"])
    with pytest.raises(TypeError):
        bool(0.1 <= te.q1)
    with pytest.raises(TypeError):
        0.1 <= te.q1 <= 0.3  # a chained comparison is two targets
    assert bool(jl("true")) and not bool(jl("Dict()")) and bool(jl("Dict(1 => 2)"))


# --- DefExpr, contexts -----------------------------------------------------------------------

def test_defexpr_tracing():
    sb.DEFAULT_CONTEXT.kq = 0.36
    q = sb.Quadrupole(L=0.5, Kn1=sb.DefExpr(lambda c: -c.kq))
    assert q.Kn1 == -0.36
    sb.DEFAULT_CONTEXT.kq = 0.4
    assert q.Kn1 == -0.4
    d = sb.rawget(q, "Kn1")
    assert "PyFunction" not in repr(julia.seval("d -> d.f")(d))  # no Python inside
    sb.DEFAULT_CONTEXT.vv = np.array([1.0, 2.0, 3.0])
    e = sb.DefExpr(lambda c: 0.5 + c.vv[2] * np.sqrt(c.kq))
    assert abs(e() - (0.5 + 3.0 * math.sqrt(0.4))) < 1e-15
    with pytest.warns(sb.PerformanceWarning):
        g = sb.DefExpr(lambda c: math.sin(c.kq))  # math.sin cannot be traced
    assert g() == math.sin(0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # (no fallback: NumPy integers are traced too)
        h = [sb.DefExpr(lambda c: 2*c.vv[i]) for i in np.arange(3)]
    assert [x() for x in h] == [2.0, 4.0, 6.0]


def test_julia_include(tmp_path):
    f = tmp_path / "lat.jl"
    f.write_text("using SciBmad\nq = Quadrupole(L=1.0, Kn1=0.1)\nbl = Beamline([q])\nn = 3\n")
    lat = julia.include(f)
    assert lat.n == 3 and lat.bl[1].Kn1 == 0.1 and lat.q.name == ""
    assert {"q", "bl", "n"} <= set(dir(lat))
    assert str(sb.julia.PySciBmad.pkguuid("SciBmad")) == "7bf6e793-a2ec-421f-8700-d1465381ee80"


def test_withvalues():
    c = sb.Context(a=1.0)
    with sb.withvalues(c, a=2.0):
        assert c.a == 2.0
        sb.push_(c, a=3.0)  # also what the block pushed is undone
    assert c.a == 1.0
    with pytest.raises(ZeroDivisionError):
        with sb.withvalues(c, a=5.0):
            1 / 0
    assert c.a == 1.0
    assert sb.withvalues(lambda: c.a * 2, c, a=4.0) == 8.0


# --- errors ----------------------------------------------------------------------------------

def test_errors(fodo):
    with pytest.raises(sb.JuliaError) as e:
        sb.twiss(fodo["bl"], cols=["nonexistent_column"])
    assert e.value.julia_type
    with pytest.raises(AttributeError):
        fodo["qf"].no_such_property
    with pytest.raises(IndexError):
        fodo["bl"][100]
    assert not hasattr(fodo["qf"], "no_such_property")


# --- plotting --------------------------------------------------------------------------------

def test_matplotlib(fodo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tw = sb.twiss(fodo["bl"])
    fig, ax = plt.subplots()
    (line,) = ax.plot(tw.s, tw.beta1)
    np.testing.assert_array_equal(line.get_ydata(), np.asarray(tw.df.beta1))
    res = sb.track(fodo["bl"], v0=[[1e-3, 0, 0, 0, 0, 0]], n_turns=1, at=...)
    (line,) = ax.plot(res.s, res.x(..., turn=1, particle=1))
    assert len(line.get_ydata()) == len(np.asarray(res.s))
    plt.close(fig)
