#!/usr/bin/env bash
# Convert an anemoi zarr dataset to a lossily compressed copy, processing
# blocks of dates in parallel.
#
#   convert_dataset.sh [options] SRC.zarr DST.zarr CONFIG.(yaml|json)
#
# Options:
#   -j N          parallel jobs (default: number of CPUs)
#   -b N          date chunks per job process (default: about 4 blocks per job;
#                 each process pays the import and numba compile cost once)
#   -v            verify: decode every chunk again and record error statistics
#   -f            overwrite chunks already present in DST (default: resume)
#   -p PYTHON     python interpreter with numcodecs, anemoi_lossy and the codec
#                 packages named in CONFIG installed (default: $PYTHON or python3)
#
# Everything except the chunks of 'data' is copied from SRC unchanged
# (dates, statistics, attributes). Re-running the same command resumes.

set -euo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

JOBS=$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 4)
BLOCK=""
VERIFY=""
OVERWRITE=""
PY=${PYTHON:-python3}

while getopts "j:b:vfp:h" opt; do
    case $opt in
        j) JOBS=$OPTARG ;;
        b) BLOCK=$OPTARG ;;
        v) VERIFY="--verify" ;;
        f) OVERWRITE="--overwrite" ;;
        p) PY=$OPTARG ;;
        h) usage 0 ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 3 ]] || usage

SRC=${1%/}
DST=${2%/}
CONFIG=$3
[[ -d $SRC ]] || { echo "error: $SRC is not a directory" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "error: $CONFIG not found" >&2; exit 1; }

# Each job is one process; keep codec libraries from spawning their own thread pools.
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BLOSC_NTHREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[$(date +%T)] preparing $DST"
NCHUNKS=$("$PY" -m anemoi_lossy.convert prepare "$SRC" "$DST" "$CONFIG")
[[ -n $BLOCK ]] || BLOCK=$(( (NCHUNKS + 4 * JOBS - 1) / (4 * JOBS) ))
NBLOCKS=$(( (NCHUNKS + BLOCK - 1) / BLOCK ))
echo "[$(date +%T)] $NCHUNKS date chunks -> $NBLOCKS blocks of $BLOCK, $JOBS parallel jobs"

# a failing block makes xargs exit non-zero; set -e then stops before finalize
seq 0 "$BLOCK" $((NCHUNKS - 1)) |
    xargs -P "$JOBS" -I{} sh -c '
        start=$1; end=$((start + $2))
        "$3" -m anemoi_lossy.convert convert "$4" "$5" "$start" "$end" $6 $7 \
            || { echo "block $start-$end failed" >&2; exit 255; }
        echo "[$(date +%T)] done dates $start-$((end - 1))"
    ' _ {} "$BLOCK" "$PY" "$SRC" "$DST" "$VERIFY" "$OVERWRITE"

echo "[$(date +%T)] finalizing"
"$PY" -m anemoi_lossy.convert finalize "$SRC" "$DST"
