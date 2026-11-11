"""Check the verified errors of a converted dataset against the bounds of the
config it was made with (written by make_challenge_config.py).

    python check_bounds.py CONFIG.yaml DST.zarr

Absolute bounds are checked against the maximum pointwise error, and lossless
variables against zero error. Relative bounds are checked against the maximum
pointwise relative error, and a value of zero must stay zero. A summary made
before the relative error was recorded cannot check relative bounds; these are
listed. NaN points are not in the errors (convert.py skips them). A NaN
that is added or removed (nan_mismatch) is a failure for all kinds.
Exits non-zero if a bound is exceeded.
"""

import json
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
summary = json.loads((Path(sys.argv[2]) / ".lossy" / "summary.json").read_text())
if "variables" not in summary:
    sys.exit("summary has no error statistics: convert with -v")

failed = 0
print(f"{'variable':<16}{'kind':<10}{'abs bound':>12}{'max_abs_err':>14}{'rel bound':>12}{'max_rel_err':>14}  status")
for name, b in config["bounds"].items():
    v = summary["variables"][name]
    err = v["max_abs_error"]
    rel = v.get("max_rel_error", float("nan"))
    rel_known = rel == rel and "zeros_changed" in v  # not NaN: every chunk recorded it
    eb_abs = {"lossless": 0.0, "abs": b.get("eb"), "abs+rel": b.get("eb")}.get(b["kind"])
    eb_rel = {"rel": b.get("eb"), "abs+rel": b.get("eb_rel")}.get(b["kind"])
    checks = []
    if eb_abs is not None:
        checks.append(err <= eb_abs)
    if eb_rel is not None:
        checks.append(rel <= eb_rel and v["zeros_changed"] == 0 if rel_known else None)
    nan_mismatch = v.get("nan_mismatch", 0)
    ok = None if None in checks else all(checks)
    if nan_mismatch:
        ok = False
    status = "relative, not recorded" if ok is None else ("ok" if ok else "EXCEEDED")
    if nan_mismatch:
        status = f"NaN MISMATCH ({nan_mismatch} points)"
    failed += ok is False
    fmt = lambda x: f"{x:>12.4g}" if x is not None else f"{'-':>12}"
    print(f"{name:<16}{b['kind']:<10}{fmt(eb_abs)}{err:>14.4g}{fmt(eb_rel)}{rel:>14.4g}  {status}")
print(f"ratio vs raw {summary['ratio_vs_raw']:.2f}x, vs source {summary['ratio_vs_source']:.2f}x")
sys.exit(1 if failed else 0)
