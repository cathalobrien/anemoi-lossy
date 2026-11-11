"""Build a convert_dataset.sh config for an anemoi dataset from the
compression-challenge requirements and codec workflows.

    python make_challenge_config.py DATASET.zarr OUT.yaml [--workflow W1|W2|W4|best|fast] [--scale 1.0]
        [--zstd-level 19] [--work-dtype float64] [--challenge-dir ../compression-challenge-search]
        [--workflow-tables configs/challenge-sheet]

Needs the compression-challenge environment (its requirements.txt, with the
numcodecs-mask/pw-ratio pins enforced) since configs are built with its
search.wrap and codec families.

Per variable:
  - forcings and the dataset's constant_fields -> lossless zstd;
  - anemoi names 'x_<level>' -> pressure-level requirements of 'x', otherwise
    single-level requirements; no requirement found or a bound of 0 ->
    lossless;
  - all requirement bounds are combined with AND. Absolute and range-relative
    bounds (pointwise or mean) become one pointwise absolute bound
    eb = scale * min(bounds); range-relative bounds use the variable's range
    from the dataset statistics. The codec gets eb minus one float32 ulp of
    the variable's largest magnitude, since decoded values are rounded to
    float32. A pointwise bound <= the mean bound implies
    the mean requirement;
  - only relative bounds -> log-domain relative quantizer (W3, pw_ratio);
  - --workflow best: use the workflow with the highest ratio for the
    variable in the ERA5-Pressure/Single-workflows tabs of the challenge
    scoreboard (CSV files in --workflow-tables). For absolute bounds, the
    choice is W1, W2 or W4. For relative bounds, W3 uses the core named in
    the W3 parameters ("(ebq core)", "(interp core)", "(sperr core)"). A
    variable that is not in the tables uses --fallback;
  - --workflow fast: standard numcodecs C codecs only, for a decode speed
    near Blosc (about 0.3 s per N320 date against 88 s for 'best', at about
    25% more bytes). Absolute bounds: uniform quantizer (fixedscaleoffset,
    step 2*eb, offset = dataset minimum, so a minimum of 0 stays exact) ->
    delta along the grid -> Blosc zstd level 9 with bitshuffle. The integer
    type has 4x headroom for values outside the dataset statistics.
    Relative bounds: bitround with keepbits k such that 2**-(k+1) <= eb,
    then Blosc zstd with byte shuffle. Pointwise relative AND absolute
    bounds (e.g. w): bitround with k large enough for both, using
    2**-(k+1) * 2 * max|x| <= eb_abs (max|x| from the dataset statistics,
    with a factor 2 margin for values outside them). Absolute bounds of a
    variable that has NaNs: the same bitround, because the quantizer cannot
    store NaN and bitround keeps NaN. The variables with NaNs are found in
    the data of --nan-dates (the 'has_nans' array of some datasets marks all
    variables). Lossless: Blosc lz4 with shuffle (as
    the source). This does not need numba or the challenge codec packages
    to decode;
  - --work-dtype: each lossy chain starts with an astype step to this dtype.
    float64 prevents the float32 failure of interp_ctx inside pw_ratio
    ("cannot satisfy the error bound") and encodes up to 2x faster. The
    lossless chains are not cast, so their size does not change;
  - data limits -> clip, minimum 0 -> zero mask, has NaNs -> NaN mask, as in
    the challenge workflows; every chain ends in zstd (level 19 by default:
    the challenge used 22, which compresses 0.05% better on O48 data but
    encodes about 5x slower).

The chosen bound per variable is written under 'bounds' (ignored by the
converter) so the verified errors in .lossy/summary.json can be checked
against it with check_bounds.py.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numcodecs
import numpy as np
import yaml

FORCINGS = re.compile(r"^(cos_|sin_)|^insolation$")


def read_small_array(path: Path) -> np.ndarray:
    meta = json.loads((path / ".zarray").read_text())
    raw = (path / "0").read_bytes()
    buf = numcodecs.get_codec(meta["compressor"]).decode(raw) if meta["compressor"] else raw
    return np.frombuffer(buf, meta["dtype"])


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def requirement_key(name: str) -> tuple[str, str]:
    m = re.fullmatch(r"(.+)_(\d+)", name)
    return ("pressure", m.group(1)) if m else ("single", name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workflow", default="W1", choices=["W1", "W2", "W4", "best", "fast"],
                   help="W1 quantize + context mixing, W2 quantize + interpolation, W4 SPERR, "
                        "best: per variable from the scoreboard workflow tables")
    p.add_argument("--fallback", default="W2", choices=["W1", "W2", "W4"],
                   help="with --workflow best: workflow for variables not in the tables")
    p.add_argument("--workflow-tables", type=Path,
                   default=Path(__file__).resolve().parents[1] / "configs" / "challenge-sheet",
                   help="directory with era5-pressure-workflows.csv and era5-single-workflows.csv")
    p.add_argument("--work-dtype", default="float64", choices=["float64", "none"],
                   help="dtype that each lossy chain casts to before encoding")
    p.add_argument("--nan-dates", type=int, nargs="+", default=[0, -1],
                   help="date indices to read to find the variables with NaNs (--workflow fast)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every error bound")
    p.add_argument("--zstd-level", type=int, default=19, help="level of the final lossless zstd stage")
    p.add_argument("--challenge-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "compression-challenge-search")
    args = p.parse_args()

    sys.path.insert(0, str(args.challenge_dir))
    from compression_recommendations import Recommendations
    from reqs import analyse
    from search import _ebq_residuals, _interp_abs, _pwr, _sperr_pwe, wrap

    lossless = {"id": "zstd.rs", "level": args.zstd_level}
    if args.workflow == "fast":
        lossless = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}
    cores = {"W1": _ebq_residuals, "W2": _interp_abs, "W4": _sperr_pwe}
    w3_cores = {"ebq": _ebq_residuals, "interp": _interp_abs, "sperr": _sperr_pwe}

    tables = {}
    if args.workflow == "best":
        for lk in ("pressure", "single"):
            with open(args.workflow_tables / f"era5-{lk}-workflows.csv", newline="") as f:
                tables[lk] = {r["Variable"]: r for r in csv.DictReader(f)}

    def choose(lk, short, relative_only):
        """Return (workflow name, core factory) for one variable."""
        if args.workflow != "best":
            return args.workflow, cores[args.workflow]
        row = tables[lk].get(short)
        if row is None:
            return f"{args.fallback} (fallback)", cores[args.fallback]
        if relative_only:
            m = re.search(r"\((\w+) core\)", row["W3 parameters"])
            name = m.group(1) if m and m.group(1) in w3_cores else "ebq"
            return f"W3 ({name} core)", w3_cores[name]
        ratio = {w: float(row[k]) for w, k in (("W1", "W1 Quantize+Context ratio"),
                                                ("W2", "W2 Quantize+Interp ratio"),
                                                ("W4", "W4 SPERR ratio")) if row[k]}
        best = max(ratio, key=ratio.get)
        return best, cores[best]

    attrs = json.loads((args.dataset / ".zattrs").read_text())
    variables = attrs["variables"]
    constants = set(attrs.get("constant_fields", []))
    # newer datasets set 'variables_with_nans' to a bool; the 'has_nans' array then gives the variables
    with_nans = attrs.get("variables_with_nans", [])
    if not isinstance(with_nans, list):
        has_nans = read_small_array(args.dataset / "has_nans") if (args.dataset / "has_nans").exists() else []
        with_nans = [v for v, n in zip(variables, has_nans) if n] if with_nans else []
    with_nans = set(with_nans)
    if args.workflow == "fast":
        # the fast quantizer cannot store NaN: find the variables with NaNs in the data
        import zarr

        data = zarr.open(str(args.dataset), mode="r")["data"]
        found = set()
        for t in args.nan_dates:
            x = data[t, :, 0, :]
            found |= {v for v, row in zip(variables, x) if np.isnan(row).any()}
        print(f"variables with NaNs in dates {args.nan_dates}: {sorted(found)}", file=sys.stderr)
        with_nans = found
    minimum, maximum = read_small_array(args.dataset / "minimum"), read_small_array(args.dataset / "maximum")
    ranges = maximum - minimum
    # decoded values are rounded to the stored dtype, which adds up to half an ulp
    dtype = json.loads((args.dataset / "data" / ".zarray").read_text())["dtype"]
    ulp = np.spacing(np.maximum(np.abs(minimum), np.abs(maximum)).astype(np.float32)).astype(np.float64)

    def fast_chain(i, name, absolute, relative, lk, short, max_rel=()):
        """Return (codec chain, bound record) for --workflow fast."""
        req = f"{lk} {short}"
        if absolute and (max_rel or name in with_nans):
            # the challenge requirements are AND: meet both bounds pointwise.
            # A variable with NaNs also uses bitround, since bitround keeps NaN.
            eb_abs = args.scale * min(absolute)
            amax = 2.0 * max(abs(float(minimum[i])), abs(float(maximum[i])))
            k_abs = int(np.ceil(np.log2(amax / eb_abs))) - 1
            chain_name = "fast bitround" + (" (NaN)" if name in with_nans else "")
            if not max_rel:
                keepbits = min(23, max(1, k_abs))
                chain = [
                    {"id": "bitround", "keepbits": keepbits},
                    {"id": "blosc", "cname": "zstd", "clevel": 9, "shuffle": 1, "blocksize": 0},
                ]
                return chain, {"kind": "abs", "eb": eb_abs, "requirement": req,
                               "workflow": f"{chain_name} {keepbits}"}
            eb_rel = args.scale * min(max_rel)
            k_rel = int(np.ceil(-np.log2(eb_rel))) - 1
            keepbits = min(23, max(k_rel, k_abs))
            chain = [
                {"id": "bitround", "keepbits": keepbits},
                {"id": "blosc", "cname": "zstd", "clevel": 9, "shuffle": 1, "blocksize": 0},
            ]
            return chain, {"kind": "abs+rel", "eb": eb_abs, "eb_rel": eb_rel, "requirement": req,
                           "workflow": f"{chain_name} {keepbits}"}
        if absolute:
            eb = args.scale * min(absolute)
            step = 2.0 * (eb - float(ulp[i]))
            offset = float(minimum[i])
            levels = 4.0 * (float(maximum[i]) - offset) / step
            int_dt = next((d for d in ("<i1", "<i2", "<i4") if 2 * levels < np.iinfo(d).max), "<i8")
            chain = [
                {"id": "astype", "encode_dtype": "<f8", "decode_dtype": np.dtype(dtype).str},
                {"id": "fixedscaleoffset", "offset": offset, "scale": 1.0 / step, "dtype": "<f8", "astype": int_dt},
                {"id": "delta", "dtype": int_dt},
                {"id": "blosc", "cname": "zstd", "clevel": 9, "shuffle": 2, "blocksize": 0},
            ]
            return chain, {"kind": "abs", "eb": eb, "requirement": req, "workflow": f"fast quantize {int_dt}"}
        eb = args.scale * min(relative)
        keepbits = int(np.ceil(-np.log2(eb))) - 1
        chain = [
            {"id": "bitround", "keepbits": keepbits},
            {"id": "blosc", "cname": "zstd", "clevel": 9, "shuffle": 1, "blocksize": 0},
        ]
        return chain, {"kind": "rel", "eb": eb, "requirement": req, "workflow": f"fast bitround {keepbits}"}

    codecs, bounds = {}, {}
    for i, name in enumerate(variables):
        if name in constants or FORCINGS.search(name):
            codecs[name] = lossless
            bounds[name] = {"kind": "lossless", "reason": "constant" if name in constants else "forcing"}
            continue
        lk, short = requirement_key(name)
        try:
            reqs = Recommendations.provide.search(markers={"grib-short-name": short, "level-kind": lk})
        except KeyError:  # the package raises KeyError for a variable it does not know
            reqs = None
        if not reqs:
            print(f"warning: no requirement for {name} ({lk} {short}), keeping it lossless", file=sys.stderr)
            codecs[name] = lossless
            bounds[name] = {"kind": "lossless", "reason": "no requirement"}
            continue
        info = analyse(reqs)
        if info.lossless:
            codecs[name] = lossless
            bounds[name] = {"kind": "lossless", "reason": "requirement"}
            continue

        rng = float(ranges[i])
        absolute = info.max_abs + info.mean_abs + [r * rng for r in info.max_range_rel + info.mean_range_rel]
        relative = info.max_rel + info.mean_rel
        if min(absolute + relative, default=1.0) <= 0.0:
            # a zero bound (e.g. cdww: mean absolute error 0) can only be met without loss
            print(f"warning: zero error bound for {name} ({lk} {short}), keeping it lossless", file=sys.stderr)
            codecs[name] = lossless
            bounds[name] = {"kind": "lossless", "reason": "zero bound"}
            continue
        if args.workflow == "fast":
            codecs[name], bounds[name] = fast_chain(i, name, absolute, relative, lk, short, info.max_rel)
            continue
        workflow, core = choose(lk, short, relative_only=not absolute)
        if absolute:
            eb = args.scale * min(absolute)
            cfg = core(eb - float(ulp[i]))
            bounds[name] = {"kind": "abs", "eb": eb, "requirement": f"{lk} {short}", "workflow": workflow}
        else:
            eb = args.scale * min(relative)
            cfg = _pwr(core)(eb)
            bounds[name] = {"kind": "rel", "eb": eb, "requirement": f"{lk} {short}", "workflow": workflow}
        cfg = wrap(cfg, info, mask_nan=name in with_nans, mask_zero=info.minimum == 0.0)
        chain = [cfg, lossless]
        if args.work_dtype != "none":
            chain.insert(0, {"id": "astype", "encode_dtype": np.dtype(args.work_dtype).str,
                             "decode_dtype": np.dtype(dtype).str})
        codecs[name] = chain

    out = {
        "grid_shape": None,
        "work_dtype": None,
        "default": lossless,
        "variables": codecs,
        "bounds": bounds,
        "generated_by": {"script": Path(__file__).name, "workflow": args.workflow, "scale": args.scale,
                         "zstd_level": args.zstd_level, "work_dtype": args.work_dtype},
    }
    args.out.write_text(yaml.dump(out, Dumper=_NoAliases, sort_keys=False, width=200))
    for name, b in bounds.items():
        print(f"{name:<16}{b['kind']:<10}{str(b.get('eb', b.get('reason'))):<24}{b.get('workflow', '')}")


if __name__ == "__main__":
    main()
