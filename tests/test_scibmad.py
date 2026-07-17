"""End-to-end tests for the ``scibmad`` Python interface.

These start a real Julia session (via ``juliacall``) and exercise the SciBmad
functionality, so they are slow on first run (Julia precompilation). They mirror
the examples in SciBmad.jl issue #76.
"""

import pytest

import scibmad as sb


# --- operator overloading (the core issue #76 fix) --------------------------

def test_defexpr_plus_number():
    e = sb.DefExpr(lambda: 1) + 10
    assert isinstance(e, sb.Operand)
    assert e() == pytest.approx(11.0)


def test_number_plus_defexpr():
    e = 10 + sb.DefExpr(lambda: 1)
    assert e() == pytest.approx(11.0)


def test_defexpr_plus_defexpr():
    e = sb.DefExpr(lambda: 1) + sb.DefExpr(lambda: 2)
    assert e() == pytest.approx(3.0)


def test_defexpr_all_binops():
    a = sb.DefExpr(lambda: 6.0)
    assert (a - 1)() == pytest.approx(5.0)
    assert (a * 2)() == pytest.approx(12.0)
    assert (a / 2)() == pytest.approx(3.0)
    assert (a ** 2)() == pytest.approx(36.0)
    assert (2 * a)() == pytest.approx(12.0)
    assert (12 / a)() == pytest.approx(2.0)


def test_defexpr_unary_minus():
    a = sb.DefExpr(lambda: 3.0)
    assert (-a)() == pytest.approx(-3.0)


def test_defexpr_math_function():
    a = sb.DefExpr(lambda: 0.0)
    assert sb.sin(a)() == pytest.approx(0.0)
    assert sb.cos(a)() == pytest.approx(1.0)


def test_defexpr_from_value():
    a = sb.DefExpr(0.36)
    assert a() == pytest.approx(0.36)


# --- context-capturing deferred expressions ---------------------------------

def test_defexpr_with_context():
    # A 1-argument lambda receives a Context; evaluate against one.
    d = sb.DefExpr(lambda c: c.a + c.b)
    ctx = sb.Context(a=1.0, b=2.0)
    assert d(ctx) == pytest.approx(3.0)


# --- time-dependent and batch parameters ------------------------------------

def test_time_dependent_param_arithmetic():
    t = sb.Time()
    p = 2.0 * t + 1.0
    assert isinstance(p, sb.Operand)
    assert p(3.0) == pytest.approx(7.0)


def test_batch_param_arithmetic():
    b = sb.BatchParam([0.1, 0.2, 0.3])
    b2 = b + 1.0
    assert isinstance(b2, sb.Operand)


# --- elements / beamline / twiss --------------------------------------------

def _make_beamline():
    E = sb.elements(
        qf=sb.Quadrupole(Kn1=0.36, L=0.5),
        d=sb.Drift(L=1.0),
        qd=sb.Quadrupole(Kn1=-0.36, L=0.5),
    )
    bl = sb.Beamline(
        [E.qf, E.d, E.qd, E.d],
        species_ref=sb.Species("electron"),
        E_ref=18e9,
    )
    return E, bl


def test_elements_sets_names():
    E, _ = _make_beamline()
    assert str(E.qf.name) == "qf"
    assert str(E.qd.name) == "qd"
    assert str(E.d.name) == "d"


def test_twiss_default():
    _, bl = _make_beamline()
    tw = sb.twiss(bl)
    # tunes accessible on the returned Twiss struct
    assert tw.tunes is not None


def test_twiss_at_empty_list():
    _, bl = _make_beamline()
    tw = sb.twiss(bl, at=[])
    assert tw is not None


def test_quadrupole_with_defexpr_strength():
    k = sb.DefExpr(lambda c: c.k1)
    qf = sb.Quadrupole(Kn1=k, L=0.5)
    qd = sb.Quadrupole(Kn1=-k, L=0.5)
    assert str(qf.name) == "" or qf is not None
    assert qd is not None


# --- pass-through surface ----------------------------------------------------

def test_passthrough_species():
    sp = sb.Species("electron")
    assert sp is not None


def test_attribute_error_for_unknown():
    with pytest.raises(AttributeError):
        sb.ThisIsNotASciBmadName
