"""Command line: ``python -m scibmad info|config|install``."""

from __future__ import annotations

import argparse
import json
import sys

from . import _runtime


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scibmad", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="start Julia and show which Julia and SciBmad.jl run")
    c = sub.add_parser("config", help="show or set which Julia runs SciBmad")
    c.add_argument("--julia", help="a Julia executable")
    c.add_argument("--project", help="the Julia project in which SciBmad is installed")
    c.add_argument("--threads", help="the number of Julia threads (or auto)")
    c.add_argument("--reset", action="store_true", help="use the default again")
    sub.add_parser("install", help="install (and precompile) Julia and SciBmad now")
    g = sub.add_parser("gpu", help="install the Julia package of a GPU kind")
    g.add_argument("kind", choices=["cuda", "metal", "amdgpu", "oneapi"])
    a = p.parse_args(argv)
    if a.cmd == "config":
        cfg = {} if a.reset else _runtime.read_config()
        for k in ("julia", "project", "threads"):
            v = getattr(a, k)
            if v is not None:
                cfg[k] = v
        if a.reset or any(getattr(a, k) is not None for k in ("julia", "project", "threads")):
            _runtime.write_config(cfg)
        print(f"{_runtime.config_file()}: {json.dumps(cfg)}")
        return 0
    import scibmad
    if a.cmd in ("info", "install"):
        for k, v in scibmad.info().items():
            print(f"{k:16s} {v}")
        return 0
    if a.cmd == "gpu":
        print(scibmad.device(a.kind))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
