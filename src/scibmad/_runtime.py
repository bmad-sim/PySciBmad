"""Finding and starting the Julia that runs SciBmad.

Julia is started once per Python process, on the first use of a SciBmad name (or by
:func:`scibmad.init`). Which Julia is used is decided in this order:

1. the arguments of :func:`scibmad.init`;
2. the environment variables ``SCIBMAD_JULIA`` (a Julia executable) and ``SCIBMAD_PROJECT``
   (a Julia project in which SciBmad is installed);
3. the configuration file (``python -m scibmad config --julia ...``);
4. the SciBmad distribution installed in the current (conda) environment
   (``conda install -c bmad-sim scibmad``), which comes with everything precompiled;
5. otherwise Julia and SciBmad are installed automatically (with juliapkg, into the Python
   environment) the first time, which takes a while once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

# The SciBmad release this version of the Python package is made for. With an automatic
# installation exactly this version is installed; other Julia installations are checked against
# SUPPORTED (a newer minor release may need a newer Python package).
SCIBMAD_VERSION = "0.8.0"
SUPPORTED = ((0, 8), (0, 9))  # [lower, upper) as (major, minor)

_GLUE = Path(__file__).with_name("_glue.jl")

STATE: dict = {"jl": None, "info": None}


class JuliaNotFoundError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Configuration


def config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "pyscibmad"


def cache_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "pyscibmad"


def config_file() -> Path:
    return config_dir() / "config.json"


def read_config() -> dict:
    try:
        return json.loads(config_file().read_text())
    except (OSError, ValueError):
        return {}


def write_config(cfg: dict) -> None:
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")


# ---------------------------------------------------------------------------------------------
# The SciBmad distribution


def _prefixes():
    seen = []
    for p in (sys.prefix, os.environ.get("CONDA_PREFIX")):
        if p and p not in seen:
            seen.append(p)
    return [Path(p) for p in seen]


def find_distribution(prefix=None):
    """The Julia executable of the SciBmad distribution in a conda prefix, or None."""
    prefixes = [Path(prefix)] if prefix else _prefixes()
    for p in prefixes:
        for exe in (p / "libexec" / "scibmad" / "bin" / "julia",
                    p / "sb" / "bin" / "julia.exe"):
            if exe.is_file():
                return exe
    return None


def is_distribution(julia) -> bool:
    """Whether a Julia executable is that of a SciBmad distribution (it has an AppEnv config)."""
    root = Path(julia).resolve().parent.parent
    return (root / "config").is_file() and (root / "etc" / "julia" / "startup.jl").is_file()


def _user_data() -> str:
    # As the distribution's launcher sets it (bin/scibmad and its startup.jl)
    if os.environ.get("USER_DATA"):
        return os.environ["USER_DATA"]
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(base) / "scibmad")


_PROBE = r"""
import Libdl
pc = Base.locate_package(Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
                                   "PythonCall"))
sb = Base.locate_package(Base.PkgId(Base.UUID("7bf6e793-a2ec-421f-8700-d1465381ee80"), "SciBmad"))
q(s) = s === nothing ? "null" : repr(String(s))
println("\n@@PYSCIBMAD@@{",
  "\"depot\": [", join(map(q, DEPOT_PATH), ","), "],",
  "\"load_path\": [", join(map(q, LOAD_PATH), ","), "],",
  "\"project\": ", q(something(Base.active_project(), "")), ",",
  "\"libjulia\": ", q(abspath(Libdl.dlpath("libjulia"))), ",",
  "\"bindir\": ", q(Sys.BINDIR), ",",
  "\"pythoncall\": ", q(pc === nothing ? nothing : dirname(dirname(pc))), ",",
  "\"scibmad\": ", q(sb), ",",
  "\"julia_version\": ", q(string(VERSION)), "}")
"""


def _probe(julia, args, env):
    """Runs Julia once to learn its paths (cached, keyed by the executable and its options)."""
    exe = Path(julia).resolve()
    key = json.dumps([str(exe), exe.stat().st_mtime, args,
                      {k: env.get(k) for k in ("JULIA_DEPOT_PATH", "JULIA_LOAD_PATH",
                                                "USER_DATA")}])
    cache = cache_dir() / "probe.json"
    try:
        entries = json.loads(cache.read_text())
    except (OSError, ValueError):
        entries = {}
    if key in entries:
        return entries[key]
    out = subprocess.run([str(exe), *args, "-e", _PROBE], env=env, capture_output=True,
                         text=True)
    if out.returncode != 0 or "@@PYSCIBMAD@@" not in out.stdout:
        raise JuliaNotFoundError(f"Could not run {exe}:\n{out.stderr[-2000:]}")
    info = json.loads(out.stdout.split("@@PYSCIBMAD@@", 1)[1].strip().splitlines()[0])
    entries[key] = info
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(entries))
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------------------------
# Starting Julia


def _choose(julia=None, project=None):
    """(kind, julia, project) of the Julia to start."""
    if julia or project:
        return "explicit", julia, project
    if os.environ.get("SCIBMAD_JULIA") or os.environ.get("SCIBMAD_PROJECT"):
        return "environment", os.environ.get("SCIBMAD_JULIA"), os.environ.get("SCIBMAD_PROJECT")
    cfg = read_config()
    if cfg.get("julia") or cfg.get("project"):
        return "config", cfg.get("julia"), cfg.get("project")
    if os.environ.get("PYTHON_JULIACALL_EXE") and os.environ.get("PYTHON_JULIACALL_PROJECT"):
        return ("juliacall", os.environ["PYTHON_JULIACALL_EXE"],
                os.environ["PYTHON_JULIACALL_PROJECT"])
    exe = find_distribution()
    if exe is not None:
        return "distribution", str(exe), None
    return "juliapkg", None, None


def _which_julia():
    from shutil import which
    exe = which("julia")
    if exe is None:
        raise JuliaNotFoundError("No Julia executable given and none found on the PATH")
    return exe


def prepare(julia=None, project=None, threads=None):
    """Decides which Julia runs SciBmad and sets juliacall's environment variables for it,
    without starting Julia (done at `import scibmad`, so that `import juliacall` afterwards
    starts the same Julia)."""
    if STATE["jl"] is not None or "juliacall" in sys.modules:
        return STATE.get("prepared")
    kind, julia, project = _choose(julia, project)
    threads = threads if threads is not None else (read_config().get("threads")
                                                    or os.environ.get("SCIBMAD_THREADS"))
    env = os.environ
    info = {"kind": kind}
    if threads is not None:
        env["PYTHON_JULIACALL_THREADS"] = str(threads)
    # Julia's signal handlers are needed as soon as Julia has more than one thread (with
    # threads > 1, and the threads that GPU packages such as Metal.jl start): its garbage
    # collector stops the other threads with them. Without them the process crashes (SIGBUS in
    # a GC safepoint). Julia's SIGINT handler is replaced by Python's again in start(), so that
    # Ctrl-C still raises KeyboardInterrupt.
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    # In IPython and Jupyter, scibmad itself sends Julia's output to the notebook (_core.py);
    # juliacall's IPython extension would do it a second time (and print a message)
    env.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")
    if kind != "juliapkg":
        julia = julia or _which_julia()
        if kind == "distribution" or is_distribution(julia):
            _prepare_distribution(julia, project, info)
        else:
            if project is None:
                raise JuliaNotFoundError(
                    "A Julia project with SciBmad is needed with this Julia executable: give "
                    "`project` (scibmad.init(project=...), SCIBMAD_PROJECT or "
                    "python -m scibmad config --project ...)")
            project = str(Path(project).expanduser().resolve())
            p = _probe(julia, [f"--project={project}", "--startup-file=no"], dict(env))
            if not p.get("pythoncall"):
                raise JuliaNotFoundError(
                    f"PythonCall.jl is not installed in the Julia project {project}: add it "
                    "(julia --project=... -e 'using Pkg; Pkg.add(\"PythonCall\")')")
            if not p.get("scibmad"):
                raise JuliaNotFoundError(f"SciBmad.jl is not installed in the Julia project "
                                         f"{project}")
            env["PYTHON_JULIACALL_EXE"] = str(julia)
            env["PYTHON_JULIACALL_PROJECT"] = project
            env["PYTHON_JULIACALL_LIB"] = p["libjulia"]
            env["PYTHON_JULIACALL_BINDIR"] = p["bindir"]
            _use_juliacall_of(p)
            info.update(julia=str(julia), project=project)
    else:
        for k in ("PYTHON_JULIACALL_EXE", "PYTHON_JULIACALL_PROJECT", "PYTHON_JULIACALL_LIB",
                  "PYTHON_JULIACALL_BINDIR"):
            env.pop(k, None)
    STATE["prepared"] = info
    return info


def start(julia=None, project=None, threads=None, quiet=False):
    """Starts Julia (once) and loads SciBmad. Returns juliacall's ``Main``."""
    if STATE["jl"] is not None:
        if julia or project:
            warnings.warn("Julia is already running; scibmad.init(julia=..., project=...) can "
                          "only take effect before the first use of SciBmad", stacklevel=3)
        return STATE["jl"]
    if STATE.get("prepare_error") is not None and not (julia or project):
        raise STATE["prepare_error"]
    if "juliacall" in sys.modules:
        info = dict(STATE.get("prepared") or {"kind": "running"})
    else:
        if julia or project or threads is not None:
            STATE["prepared"] = None
        info = prepare(julia, project, threads) if STATE.get("prepared") is None else \
            STATE["prepared"]
        info = dict(info)
        if info["kind"] == "juliapkg" and not quiet:
            _note_juliapkg()
    sigint = _python_sigint()
    import juliacall  # noqa: E402  (starts Julia)
    _restore_sigint(sigint)
    jl = juliacall.Main
    if info["kind"] == "juliapkg":
        import juliapkg
        info.update(julia=juliapkg.executable(), project=juliapkg.project())
    try:
        jl.seval("using SciBmad")
    except Exception as e:
        raise JuliaNotFoundError(f"SciBmad.jl could not be loaded in this Julia ({info}): {e}")
    jl.seval(f"Base.include(Main, {json.dumps(str(_GLUE))})")
    glue = jl.PySciBmad
    info["scibmad_version"] = str(glue.version(jl.SciBmad))
    info["julia_version"] = str(jl.seval("string(VERSION)"))
    info["threads"] = int(jl.seval("Threads.nthreads()"))
    STATE["jl"] = jl
    STATE["info"] = info
    _check_version(info["scibmad_version"])
    return jl


def _python_sigint():
    import signal
    import threading
    if threading.current_thread() is not threading.main_thread():
        return None
    return signal.getsignal(signal.SIGINT)


def _restore_sigint(handler):
    """With Julia's signal handlers, Julia's SIGINT handler ends the process on Ctrl-C: put back
    Python's, so that Ctrl-C raises KeyboardInterrupt (when Julia returns to Python)."""
    if handler is None or os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes":
        return
    import signal
    signal.signal(signal.SIGINT, handler)


def _prepare_distribution(julia, project, info):
    """Environment for the SciBmad distribution, as its launcher and startup file set it up:
    its own depot and load path (the bundled packages), a per-user depot (USER_DATA) and the
    juliacall Python package that belongs to its PythonCall."""
    env = os.environ
    user_data = _user_data()
    Path(user_data).mkdir(parents=True, exist_ok=True)
    env["USER_DATA"] = user_data
    p = _probe(julia, ["--startup-file=yes"], dict(env))
    sep = ";" if os.name == "nt" else ":"
    env["JULIA_DEPOT_PATH"] = sep.join(p["depot"])
    env["JULIA_LOAD_PATH"] = sep.join(p["load_path"])
    project = str(Path(project).expanduser().resolve()) if project else user_data
    if not Path(project, "Project.toml").exists() and project == user_data:
        Path(project, "Project.toml").touch()
    env["PYTHON_JULIACALL_EXE"] = str(julia)
    env["PYTHON_JULIACALL_PROJECT"] = project
    env["PYTHON_JULIACALL_LIB"] = p["libjulia"]
    env["PYTHON_JULIACALL_BINDIR"] = p["bindir"]
    _use_juliacall_of(p)
    info.update(distribution=True, julia=str(julia), project=project)


def _use_juliacall_of(p):
    """The Python package juliacall must be the version of the Julia package PythonCall: use the
    copy that comes with PythonCall (its `pysrc`), whatever juliacall pip installed."""
    if p.get("pythoncall"):
        pysrc = str(Path(p["pythoncall"]) / "pysrc")
        if Path(pysrc, "juliacall", "__init__.py").exists() and pysrc not in sys.path:
            sys.path.insert(0, pysrc)


def _note_juliapkg():
    try:
        import juliapkg
        juliapkg.project()  # fast if already resolved
        resolved = Path(juliapkg.project(), "Manifest.toml").exists()
    except Exception:
        resolved = False
    if not resolved:
        print("scibmad: installing Julia and SciBmad.jl for this Python environment (only the "
              "first time; this can take a long time). Faster: conda install -c bmad-sim "
              "pyscibmad, which comes with everything precompiled.", file=sys.stderr)


def _check_version(v):
    try:
        major, minor = (int(x) for x in v.split(".")[:2])
    except ValueError:
        return
    lo, hi = SUPPORTED
    if not (lo <= (major, minor) < hi):
        warnings.warn(f"SciBmad.jl {v} is outside the versions this scibmad Python package "
                      f"supports ({lo[0]}.{lo[1]} up to, not including, {hi[0]}.{hi[1]}); "
                      "update the Python package (pip install -U pyscibmad)", stacklevel=4)


def info() -> dict:
    """Which Julia and SciBmad are running (starts them if needed)."""
    start()
    return dict(STATE["info"])
