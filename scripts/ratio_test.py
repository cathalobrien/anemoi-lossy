"""Measure the compression of a converter config on an anemoi dataset,
without writing a new dataset.

    python ratio_test.py DATASET.zarr CONFIG.yaml OUT.jsonl --dates 0 15 30 [--jobs N]

The source chunks are read as they are stored. Each chunk must hold one
variable (chunks [1, 1, 1, n]), or all variables (chunks [1, nvar, 1, n]).
The codec chain of each variable is applied to each block of n grid points.
This is the data that the converter stores per variable when the dataset is
rechunked to [1, nvar, 1, n].

For each (date, variable, block), one JSON line is written with the raw,
source and encoded sizes, the encode and decode times, and the maximum
absolute and relative errors. NaN points are not in the errors; a NaN that
is added or removed is counted in 'nan_mismatch' and is a bound violation.
The summary is printed at the end.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numcodecs
import numpy as np
import yaml
from numcodecs.compat import ensure_ndarray

ARGS = None


def _chain(config, name):
    import fnmatch

    for pat, c in config["variables"].items():
        if fnmatch.fnmatchcase(name, pat):
            return c if isinstance(c, list) else [c]
    c = config["default"]
    return c if isinstance(c, list) else [c]


def _work(task):
    t, v = task
    src, meta, attrs, config = ARGS
    name = attrs["variables"][v]
    codecs = [numcodecs.get_codec(dict(c)) for c in _chain(config, name)]
    source = numcodecs.get_codec(meta["compressor"])
    n_blocks = -(-meta["shape"][3] // meta["chunks"][3])
    per_var_chunk = meta["chunks"][1] == 1
    out = []
    for b in range(n_blocks):
        if per_var_chunk:
            raw = (src / "data" / f"{t}.{v}.0.{b}").read_bytes()
            x = np.frombuffer(source.decode(raw), meta["dtype"]).copy()
            src_bytes = len(raw)
        else:
            raw = (src / "data" / f"{t}.0.0.{b}").read_bytes()
            a = np.frombuffer(source.decode(raw), meta["dtype"]).reshape(meta["chunks"][1], -1)
            x = np.ascontiguousarray(a[v])
            src_bytes = len(raw) * x.size / a.size  # share of the chunk, approximate
        t0 = time.perf_counter()
        buf = x
        for c in codecs:  # as PerVariable: pass arrays between steps
            buf = c.encode(buf)
            if isinstance(buf, (bytes, bytearray, memoryview)):
                buf = np.frombuffer(buf, "u1")
        enc = ensure_ndarray(buf).tobytes()
        te = time.perf_counter() - t0
        t0 = time.perf_counter()
        y = np.frombuffer(enc, "u1")
        for c in reversed(codecs):
            y = c.decode(y)
            if isinstance(y, (bytes, bytearray, memoryview)):
                y = np.frombuffer(y, "u1")
        y = ensure_ndarray(y)
        y = (y if y.dtype.kind == "f" else y.view(meta["dtype"])).reshape(-1).astype(meta["dtype"])
        td = time.perf_counter() - t0
        n, nbytes = int(x.size), int(x.nbytes)
        # NaN points are not in the errors; a NaN that is added or removed is counted
        xn, yn = np.isnan(x), np.isnan(y)
        nan_mismatch = int((xn != yn).sum())
        keep = ~xn & ~yn
        x, y = x[keep], y[keep]
        d = np.abs(y.astype(np.float64) - x.astype(np.float64))
        nz = x != 0
        out.append(
            {
                "date": t, "var": name, "block": b, "n": n, "raw": nbytes,
                "src": float(src_bytes), "enc": len(enc), "t_enc": te, "t_dec": td,
                "max_abs": float(d.max()) if d.size else 0.0,
                "max_rel": float((d[nz] / np.abs(x[nz])).max()) if nz.any() else 0.0,
                "exact_zeros": bool((y[~nz] == 0).all()),
                "nan_mismatch": nan_mismatch,
            }
        )
    return out


def _init(args):
    global ARGS
    ARGS = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--dates", type=int, nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args()

    meta = json.loads((args.dataset / "data" / ".zarray").read_text())
    attrs = json.loads((args.dataset / ".zattrs").read_text())
    config = yaml.safe_load(args.config.read_text())
    tasks = [(t, v) for t in args.dates for v in range(len(attrs["variables"]))]
    # the slowest variables first, so that the pool ends at about the same time
    tasks.sort(key=lambda tv: 0 if attrs["variables"][tv[1]].startswith(("u_", "v_", "q_", "t_")) else 1)

    rows = []
    with Pool(args.jobs, initializer=_init, initargs=((args.dataset, meta, attrs, config),)) as pool, open(args.out, "w") as f:
        for i, res in enumerate(pool.imap_unordered(_work, tasks)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            rows += res
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} fields", flush=True)

    bounds = config.get("bounds", {})
    viol = []
    for r in rows:
        b = bounds.get(r["var"], {})
        if r.get("nan_mismatch", 0):
            viol.append(r)
            continue
        if b.get("kind") == "abs" and r["max_abs"] > b["eb"]:
            viol.append(r)
        if b.get("kind") == "rel" and r["max_rel"] > b["eb"]:
            viol.append(r)
        if b.get("kind") == "abs+rel" and (r["max_abs"] > b["eb"] or r["max_rel"] > b["eb_rel"]):
            viol.append(r)
        if b.get("kind") == "lossless" and r["max_abs"] != 0:
            viol.append(r)
    raw = sum(r["raw"] for r in rows)
    src = sum(r["src"] for r in rows)
    enc = sum(r["enc"] for r in rows)
    nd = len(args.dates)
    print(f"dates {args.dates}: raw {raw / nd / 1e9:.3f} GB/date, source {src / nd / 1e9:.3f} GB/date, "
          f"lossy {enc / nd / 1e9:.3f} GB/date")
    print(f"ratio vs raw {raw / enc:.2f}x, vs source {src / enc:.2f}x")
    print(f"CPU time per date: encode {sum(r['t_enc'] for r in rows) / nd:.1f} s, "
          f"decode {sum(r['t_dec'] for r in rows) / nd:.1f} s")
    print(f"bound violations: {len(viol)}")
    for r in viol[:20]:
        print("  ", r)


if __name__ == "__main__":
    main()
