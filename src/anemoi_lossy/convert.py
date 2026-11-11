"""Re-encode the ``data`` array of an anemoi zarr (v2, directory store) with a
per-variable codec. Driven by ``scripts/convert_dataset.sh``; the three steps
can also be run by hand:

    python -m anemoi_lossy.convert prepare  SRC DST CONFIG   # prints number of date chunks
    python -m anemoi_lossy.convert convert  SRC DST START END [--verify] [--overwrite]
    python -m anemoi_lossy.convert finalize SRC DST

Works on raw chunk files, so it needs numcodecs (and the codec packages named
in CONFIG) but not zarr. Every chunk is written to a temporary file and renamed,
so an interrupted run can be resumed. Per-chunk sizes and (with --verify) error
statistics go to ``DST/.lossy/stats-*.jsonl``; ``finalize`` aggregates them.

The chunks of ``data`` can contain all variables and grid points of a date
(the anemoi-datasets default), or only some variables and some grid points.
The new dataset has the same chunks as the source. The converter gives the
first variable of each chunk to the codec (``PerVariable.encode_block``), so
the codec can apply the correct chain.
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from numcodecs.compat import ensure_ndarray
from numcodecs.registry import get_codec

from anemoi_lossy.codec import PerVariable

WORK = ".lossy"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=True))
    os.replace(tmp, path)


def _load_config(path: Path) -> dict:
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(path.read_text())
    return _read_json(path)


def _chunk_key(meta: dict, idx: tuple[int, ...]) -> str:
    return meta.get("dimension_separator", ".").join(str(i) for i in idx)


def _check_layout(meta: dict) -> None:
    shape, chunks = meta["shape"], meta["chunks"]
    if len(shape) != 4:
        raise SystemExit(f"expected data of shape (dates, variables, ensembles, values), got {shape}")
    if meta.get("order", "C") != "C":
        raise SystemExit("only C-order arrays are supported")


def _n_chunks(meta: dict) -> list[int]:
    """Number of chunks along each axis (dates, variables, ensembles, values)."""
    return [-(-s // c) for s, c in zip(meta["shape"], meta["chunks"])]


def build_codec_config(config: dict, variables: list[str], chunk_shape: list[int]) -> dict:
    """Resolve the user config (default + per-variable fnmatch patterns) into a
    PerVariable config with one entry per unique codec chain."""
    patterns = config.get("variables") or {}
    if "default" not in config:
        raise SystemExit("config needs a 'default' codec")

    unique, index = [], []
    for name in variables:
        chain = next((c for pat, c in patterns.items() if fnmatch.fnmatchcase(name, pat)), config["default"])
        chain = chain if isinstance(chain, list) else [chain]
        if chain not in unique:
            unique.append(chain)
        index.append(unique.index(chain))

    cfg = {
        "id": PerVariable.codec_id,
        "chunk_shape": chunk_shape,
        "codecs": unique,
        "index": index,
        "grid_shape": config.get("grid_shape"),
        "work_dtype": config.get("work_dtype"),
    }
    get_codec(cfg)  # fail early on unknown codecs or a bad grid_shape
    return cfg


def prepare(src: Path, dst: Path, config_path: Path) -> None:
    if (src / "zarr.json").exists():
        raise SystemExit("zarr v3 stores are not supported yet")
    meta = _read_json(src / "data" / ".zarray")
    _check_layout(meta)
    variables = _read_json(src / ".zattrs")["variables"]

    if dst.exists() and not (dst / WORK).exists():
        raise SystemExit(f"{dst} exists and was not created by this tool")

    # everything except the chunks of 'data' is copied verbatim
    def ignore(directory, names):
        if Path(directory).resolve() == (src / "data").resolve():
            return [n for n in names if not n.startswith(".")]
        return []

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    (dst / WORK).mkdir(exist_ok=True)

    config = _load_config(config_path)
    codec_cfg = build_codec_config(config, variables, meta["chunks"])

    new_meta = dict(meta, compressor=codec_cfg, filters=None)
    _write_json(dst / "data" / ".zarray", new_meta)
    if (dst / ".zmetadata").exists():
        zmeta = _read_json(dst / ".zmetadata")
        zmeta["metadata"]["data/.zarray"] = new_meta
        _write_json(dst / ".zmetadata", zmeta)

    _write_json(dst / WORK / "source.json", {"source": str(src.resolve()), "config": config, "source_meta": meta})
    print(_n_chunks(meta)[0])


def _source_codecs(meta: dict):
    compressor = get_codec(meta["compressor"]) if meta.get("compressor") else None
    filters = [get_codec(f) for f in meta.get("filters") or []]
    return compressor, filters


def _decode_source(raw: bytes, meta: dict, compressor, filters) -> np.ndarray:
    buf = compressor.decode(raw) if compressor else raw
    for f in reversed(filters):
        buf = f.decode(buf)
    return ensure_ndarray(buf).view(meta["dtype"]).reshape(meta["chunks"])


def _error_stats(orig: np.ndarray, recon: np.ndarray) -> list[dict]:
    stats = []
    for i in range(orig.shape[1]):
        o = orig[:, i].astype(np.float64)
        r = recon[:, i].astype(np.float64)
        on, rn = np.isnan(o), np.isnan(r)
        ok = ~on & ~rn
        d = np.abs(o[ok] - r[ok])
        mag = np.abs(o[ok])
        nz = mag > 0
        # relative error is not defined where the original value is zero
        rel = d[nz] / mag[nz]
        stats.append(
            {
                "n": int(ok.sum()),
                "max_abs": float(d.max()) if d.size else 0.0,
                "max_rel": float(rel.max()) if rel.size else 0.0,
                "zero_changed": int((d[~nz] > 0).sum()),
                "sum_sq": float((d**2).sum()),
                "nan_mismatch": int((on != rn).sum()),
            }
        )
    return stats


def convert(src: Path, dst: Path, start: int, end: int, verify: bool, overwrite: bool) -> None:
    src_meta = _read_json(src / "data" / ".zarray")
    dst_meta = _read_json(dst / "data" / ".zarray")
    compressor, filters = _source_codecs(src_meta)
    codec = get_codec(dst_meta["compressor"])
    n_dates, n_var, n_ens, n_val = _n_chunks(src_meta)
    end = min(end, n_dates)
    shape, chunks = src_meta["shape"], src_meta["chunks"]

    chunk_nbytes = int(np.prod(chunks)) * np.dtype(src_meta["dtype"]).itemsize
    log = open(dst / WORK / f"stats-{start:08d}-{end:08d}.jsonl", "w")
    for t, v, e, g in itertools.product(range(start, end), range(n_var), range(n_ens), range(n_val)):
        key = _chunk_key(src_meta, (t, v, e, g))
        src_file, dst_file = src / "data" / key, dst / "data" / key
        if not src_file.exists():  # chunk never written: zarr returns fill_value
            continue
        if dst_file.exists() and not overwrite and not verify:  # resumed run: already done
            rec = {"chunk": key, "raw": chunk_nbytes, "src": src_file.stat().st_size, "dst": dst_file.stat().st_size}
            log.write(json.dumps(rec) + "\n")
            continue

        var0 = v * chunks[1]  # dataset index of the first variable of the chunk
        raw = src_file.read_bytes()
        arr = _decode_source(raw, src_meta, compressor, filters)
        if dst_file.exists() and not overwrite:
            enc = dst_file.read_bytes()
        else:
            enc = codec.encode_block(arr, var0)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_file.with_name(dst_file.name + f".tmp{os.getpid()}")
            tmp.write_bytes(enc)
            os.replace(tmp, dst_file)

        rec = {"chunk": key, "raw": arr.nbytes, "src": len(raw), "dst": len(enc), "var0": var0}
        if verify:
            # edge chunks extend past the array: do not use the padding in the errors
            valid = tuple(slice(0, min(c, s - i * c)) for s, c, i in zip(shape, chunks, (t, v, e, g)))
            rec["vars"] = _error_stats(arr[valid], codec.decode(enc)[valid])
        log.write(json.dumps(rec) + "\n")
    log.close()


def finalize(src: Path, dst: Path) -> None:
    variables = _read_json(dst / ".zattrs")["variables"]
    raw = src_b = dst_b = n = 0
    per_var = [{"n": 0, "max_abs": 0.0, "max_rel": 0.0, "zero_changed": 0, "sum_sq": 0.0, "nan_mismatch": 0}
               for _ in variables]
    verified = False
    for path in sorted((dst / WORK).glob("stats-*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            n += 1
            raw += rec["raw"]
            src_b += rec["src"]
            dst_b += rec["dst"]
            var0 = rec.get("var0", 0)
            for acc, s in zip(per_var[var0:], rec.get("vars", [])):
                verified = True
                acc["n"] += s["n"]
                acc["sum_sq"] += s["sum_sq"]
                acc["nan_mismatch"] += s["nan_mismatch"]
                acc["max_abs"] = max(acc["max_abs"], s["max_abs"])
                # stats written before max_rel was added have no relative error
                acc["max_rel"] = max(acc["max_rel"], s.get("max_rel", float("nan")))
                acc["zero_changed"] += s.get("zero_changed", 0)

    summary = {
        "chunks": n,
        "raw_bytes": raw,
        "source_bytes": src_b,
        "lossy_bytes": dst_b,
        "ratio_vs_raw": raw / dst_b if dst_b else None,
        "ratio_vs_source": src_b / dst_b if dst_b else None,
    }
    if verified:
        summary["variables"] = {
            name: {
                "max_abs_error": acc["max_abs"],
                "max_rel_error": acc["max_rel"],
                "zeros_changed": acc["zero_changed"],
                "rmse": (acc["sum_sq"] / acc["n"]) ** 0.5 if acc["n"] else None,
                "nan_mismatch": acc["nan_mismatch"],
            }
            for name, acc in zip(variables, per_var)
        }
    _write_json(dst / WORK / "summary.json", summary)

    attrs = _read_json(dst / ".zattrs")
    source = _read_json(dst / WORK / "source.json")
    attrs["lossy_compression"] = {"source": source["source"], "config": source["config"]}
    _write_json(dst / ".zattrs", attrs)
    if (dst / ".zmetadata").exists():
        zmeta = _read_json(dst / ".zmetadata")
        zmeta["metadata"][".zattrs"] = attrs
        _write_json(dst / ".zmetadata", zmeta)

    print(f"chunks: {n}  raw: {raw / 2**30:.3f} GiB  source: {src_b / 2**30:.3f} GiB  lossy: {dst_b / 2**30:.3f} GiB")
    if dst_b:
        print(f"ratio vs raw: {raw / dst_b:.2f}x  vs source: {src_b / dst_b:.2f}x")
    if verified:
        print(f"{'variable':<16}{'max_abs_err':>14}{'rmse':>14}{'nan_mismatch':>14}")
        for name, v in summary["variables"].items():
            print(f"{name:<16}{v['max_abs_error']:>14.4g}{v['rmse'] or 0:>14.4g}{v['nan_mismatch']:>14d}")
    print(f"summary written to {dst / WORK / 'summary.json'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("src", type=Path)
    a.add_argument("dst", type=Path)
    a.add_argument("config", type=Path)
    a = sub.add_parser("convert")
    a.add_argument("src", type=Path)
    a.add_argument("dst", type=Path)
    a.add_argument("start", type=int)
    a.add_argument("end", type=int)
    a.add_argument("--verify", action="store_true")
    a.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("finalize")
    a.add_argument("src", type=Path)
    a.add_argument("dst", type=Path)
    args = p.parse_args(argv)

    if args.cmd == "prepare":
        prepare(args.src, args.dst, args.config)
    elif args.cmd == "convert":
        convert(args.src, args.dst, args.start, args.end, args.verify, args.overwrite)
    else:
        finalize(args.src, args.dst)


if __name__ == "__main__":
    sys.exit(main())
