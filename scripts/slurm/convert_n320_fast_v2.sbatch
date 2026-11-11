#!/usr/bin/env bash
# Convert the N320 ERA5 dataset (1979-2023, 6h) to a lossy copy with the
# fast-decode codecs (configs/n320-fast-v2.yaml, --workflow fast; w meets its
# absolute and relative bounds), and verify
# each chunk. The work is divided between the nodes of the job. On each node,
# xargs runs JOBS single-threaded processes in parallel.
#
# Each process uses about 0.65 GB, so 280 processes fit in the free memory
# (about 286 GB per gpu node).
#
# Re-submit the same script to resume. Chunks that exist are not encoded
# again, but with verification each chunk is decoded again (about 0.3 s).
#
#   sbatch convert_n320_fast_v2.sbatch

#SBATCH --job-name=n320-lossy-fast-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=/lus/h1aiws01/naco/compressed_datasets/logs/%x-%j.out

set -euo pipefail

PY=/perm/naco/venvs/compressed-zarr/bin/python
LZ=/lus/h2resw01/hpcperm/naco/aifs/tests/compressed-zarr/lossy-zarr
SRC=/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8.zarr
DST=/lus/h1aiws01/naco/compressed_datasets/aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8-lossy-fast-v2.zarr
CONFIG=$LZ/configs/n320-fast-v2.yaml
JOBS=${JOBS:-280}    # processes per node
BLOCK=${BLOCK:-16}   # dates per process (about 70 s with verification)

export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BLOSC_NTHREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[$(date +%F' '%T)] job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: $JOBS jobs/node, $BLOCK dates/block"
echo "[$(date +%F' '%T)] preparing $DST"
NCHUNKS=$("$PY" -m anemoi_lossy.convert prepare "$SRC" "$DST" "$CONFIG")
echo "[$(date +%F' '%T)] $NCHUNKS date chunks"

# One task per node. Node k does the blocks with (block index mod nodes) == k.
export PY SRC DST JOBS BLOCK NCHUNKS
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    export NUMBA_CACHE_DIR=/tmp/numba-$SLURM_JOB_ID
    seq 0 "$BLOCK" $((NCHUNKS - 1)) | awk -v n="$SLURM_NTASKS" -v k="$SLURM_PROCID" "(NR - 1) % n == k" |
        xargs -P "$JOBS" -I{} sh -c "
            start=\$1; end=\$((start + $BLOCK))
            \"$PY\" -m anemoi_lossy.convert convert \"$SRC\" \"$DST\" \$start \$end --verify \
                || { echo \"block \$start-\$end failed on \$(hostname)\" >&2; exit 255; }
            echo \"[\$(date +%F\" \"%T)] \$(hostname) done dates \$start-\$((end - 1))\"
        " _ {}
    rm -rf "$NUMBA_CACHE_DIR"
'

echo "[$(date +%F' '%T)] finalizing"
"$PY" -m anemoi_lossy.convert finalize "$SRC" "$DST"
"$PY" "$(dirname "$CONFIG")/../scripts/check_bounds.py" "$CONFIG" "$DST" || echo "check_bounds reported violations"
echo "[$(date +%F' '%T)] finished"
