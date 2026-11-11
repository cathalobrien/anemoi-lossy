# anemoi-lossy

Error-bounded lossy compression of anemoi zarr datasets for AIFS training.

This package converts an anemoi dataset to a copy that is smaller. Each
variable has an error bound. The decoded values are in this bound at each grid
point. anemoi-datasets and anemoi-training read the copy with no code change.

The text of this document uses ASD-STE100 (Simplified Technical English).

## Contents

1. [Design](#1-design)
2. [Installation](#2-installation)
3. [Procedure: convert a dataset](#3-procedure-convert-a-dataset)
4. [Procedure: train on a converted dataset](#4-procedure-train-on-a-converted-dataset)
5. [Configuration files](#5-configuration-files)
6. [Current results](#6-current-results)
7. [Limits](#7-limits)
8. [Next steps](#8-next-steps)
9. [Repository layout](#9-repository-layout)

## 1. Design

### 1.1 The problem

An anemoi dataset has a `data` array with the shape
`(dates, variables, ensembles, values)`. Usually there is one chunk for each
date, with the shape `(1, n_variables, 1, n_values)`. The chunks are
compressed with Blosc LZ4. Blosc is lossless, so it makes the data only about
2 times smaller.

Zarr v2 permits only one compressor for each array. But each variable needs a
different lossy codec and a different error bound.

### 1.2 The per-variable codec

The codec `anemoi.per_variable` (`src/anemoi_lossy/codec.py`) solves this
problem. It operates as one zarr compressor:

1. It divides each chunk along the variable axis.
2. It applies one codec chain to each variable.
3. It writes all the variable payloads behind a small header.

The codec configuration is in `data/.zarray`, field `compressor`:

| Field | Meaning |
|---|---|
| `chunk_shape` | The full chunk shape of the array. |
| `codecs` | The list of unique codec chains. |
| `index` | For each variable, the index of its chain in `codecs`. |
| `grid_shape` | Optional. The 2D shape of the values axis, for codecs that use 2D data. Use `null` for reduced Gaussian grids. |
| `work_dtype` | Optional. The data type for encoding. |

The encoded chunk has this layout:

```
b"APV1" | u32 header length | JSON header (dtype, shape, payload lengths) | payloads
```

### 1.3 Decoding in training

anemoi-training does not decode data itself. Zarr decodes each chunk when a
data loader worker reads a sample. Zarr finds the codec by its `id` in the
numcodecs registry.

This package registers the codec through a `numcodecs.codecs` entry point
(`pyproject.toml`). Thus zarr finds the codec with no `import`. The only
requirement is: this package must be on the Python path of each process that
reads the data. This includes the data loader workers.

We verified this with zarr 2.18 and zarr 3.4, in a clean process
(`python -I`).

### 1.4 The converter

The converter (`src/anemoi_lossy/convert.py`) has three steps:

| Step | Command | What it does |
|---|---|---|
| prepare | `python -m anemoi_lossy.convert prepare SRC DST CONFIG` | Copies all of `SRC` except the `data` chunks. Writes the new `data/.zarray`. Prints the number of date chunks. |
| convert | `python -m anemoi_lossy.convert convert SRC DST START END [--verify] [--overwrite]` | Encodes the date chunks `START` to `END - 1`. With `--verify`, it decodes each chunk again and records the errors. |
| finalize | `python -m anemoi_lossy.convert finalize SRC DST` | Updates `.zmetadata` and `.zattrs`. Writes `DST/.lossy/summary.json`. |

Important properties:

- The converter reads and writes the raw chunk files. It needs numcodecs, but
  not zarr.
- It copies the dates, coordinates, attributes and **the statistics of the
  original data** with no change. Thus the normalisation in training does not
  change.
- It adds a `lossy_compression` entry to `.zattrs` for provenance.
- It writes each chunk to a temporary file and then renames it. If a run
  stops, run it again. It continues from the last chunk.
- It writes the sizes and errors of each chunk to `DST/.lossy/stats-*.jsonl`.
- The errors do not include NaN points. The converter counts each NaN that
  the codec adds or removes (`nan_mismatch`). A mismatch is a failure.

The converter accepts only:

- zarr v2 directory stores;
- chunks that contain all variables and all grid points of a date;
- C order.

It rejects zarr v3 stores.

### 1.5 Codec chains

The script `scripts/make_challenge_config.py` makes a converter configuration
for a dataset. It gets the error bounds from the requirements of the ERA5
compression challenge (package `compression-recommendations`, version
0.1.0b1).

Rules for each variable:

- A name `x_<level>` gets the pressure-level requirement of `x`. Other names
  get the single-level requirement.
- Forcings and the `constant_fields` of the dataset stay lossless.
- A variable with no requirement, or with a bound of 0, stays lossless.
- All the bounds of a variable apply together (AND). Absolute and
  range-relative bounds become one pointwise absolute bound. A mean bound
  becomes a pointwise bound. This is stricter than necessary.
- The codec gets the bound minus one float32 ulp of the largest value of the
  variable. This is necessary because the decoder rounds the values to
  float32.

The workflow `fast` (`--workflow fast`) uses only standard numcodecs codecs
written in C. Thus it decodes almost as fast as Blosc. It has four chain types:

| Bound type | Codec chain |
|---|---|
| Absolute | `astype` float64 → `fixedscaleoffset` (step 2 × bound, offset = dataset minimum) → `delta` → Blosc zstd level 9 with bit shuffle. The integer type has 4 times headroom for values outside the dataset statistics. A minimum of 0 stays exact. |
| Relative | `bitround` with `k` mantissa bits, so that 2^-(k+1) ≤ bound → Blosc zstd with byte shuffle. |
| Absolute AND relative (for example `w`) | `bitround` with enough bits for the two bounds. The absolute bound uses 2 × the largest magnitude in the statistics, as a safety margin. |
| Absolute, variable has NaN | `bitround`, because the quantizer cannot keep NaN. The script finds NaN in the data of `--nan-dates`. |
| Lossless | Blosc LZ4 with byte shuffle (the same as the source). |

The other workflows (`W1`, `W2`, `W4`, `best`) use the codecs of the
compression challenge: quantization with context mixing, quantization with
interpolation, log-domain relative quantization and SPERR. They give a higher
ratio, but they decode approximately 400 times slower (Section 6).

## 2. Installation

### 2.1 Converter and training

The converter and the codec need only numpy, numcodecs and pyyaml.

```bash
python3 -m venv venv
venv/bin/pip install numpy pyyaml numcodecs
venv/bin/pip install -e '.[test]'
venv/bin/python -m pytest tests -q
```

Use the same numcodecs version in the converter environment and in the
training environment. We used numcodecs 0.15.1.

Some training environments remove packages that are not in their lock file
(for example `uvm sync` in aifs-dev). For these environments, install the
package into a separate directory and add it to `PYTHONPATH`:

```bash
venv/bin/pip install --no-deps --target $PWD/anemoi-lossy-site .
export PYTHONPATH=$PWD/anemoi-lossy-site${PYTHONPATH:+:$PYTHONPATH}
```

### 2.2 Configuration generator

`scripts/make_challenge_config.py` needs the environment of the compression
challenge (repository `compression-challenge-search`). By default, the script
looks for it next to this repository. Use `--challenge-dir` to give a
different path.

To make this environment:

1. Install the `requirements.txt` of `compression-challenge-search`.
2. Use a uv override that pins `numcodecs-mask` and `numcodecs-pw-ratio` to
   the commits in `requirements.txt`. If you do not do this, the resolution
   fails.
3. Run `pip install --no-deps compression-requirement-checks==0.1.0a2`.

You need this environment only to make configurations. You do not need it to
convert or read a dataset with the workflow `fast`.

## 3. Procedure: convert a dataset

### 3.1 Make the configuration

```bash
python scripts/make_challenge_config.py SRC.zarr configs/my-config.yaml --workflow fast
```

Important options:

| Option | Default | Meaning |
|---|---|---|
| `--workflow` | `W1` | `fast`, `W1`, `W2`, `W4` or `best`. Use `fast` for training. |
| `--scale` | `1.0` | Multiplies all error bounds. |
| `--nan-dates` | `0 -1` | The dates that the script reads to find variables with NaN. |
| `--zstd-level` | `19` | The level of the last zstd step (not for `fast`). |

The quantizer offsets come from the statistics of the dataset. **Do not use a
configuration with a different dataset.** Write the dataset name and uuid at
the top of the configuration file.

### 3.2 Measure the ratio on some dates

Do this test before a full conversion. It does not write a dataset.

```bash
python scripts/ratio_test.py SRC.zarr configs/my-config.yaml ratio.jsonl --dates 0 33600 67200 --jobs 64
```

The script prints the size per date, the ratio, the decode time and the
number of bound violations. If there are violations, do not continue.

### 3.3 Convert the dataset

On one machine:

```bash
scripts/convert_dataset.sh -v -j 64 SRC.zarr DST.zarr configs/my-config.yaml
```

| Option | Meaning |
|---|---|
| `-j N` | Number of parallel processes. Default: number of CPUs. |
| `-b N` | Dates for each process. Default: approximately 4 blocks for each process. |
| `-v` | Decode each chunk again and record the errors. Always use this option. |
| `-f` | Overwrite the chunks that exist. Default: continue from the last chunk. |
| `-p PYTHON` | The Python interpreter. Default: `$PYTHON` or `python3`. |

Each process is single-threaded. If one block fails, the script stops before
`finalize`.

On a Slurm cluster, use a script in `scripts/slurm/` as a template. These
scripts divide the dates between the nodes of the job. Change these items:

- the `#SBATCH` lines (partition, account, time, log path);
- `PY`, `SRC`, `DST` and `CONFIG`;
- `JOBS`: each process uses approximately 0.65 GB of memory. On a node with
  approximately 290 GB of free memory, use 280 processes.

To continue after a failure, submit the same script again.

### 3.4 Check the result

```bash
python scripts/check_bounds.py configs/my-config.yaml DST.zarr
```

Each variable must show `ok`. The script exits with an error if a bound is
exceeded or if there is a NaN mismatch. Also read `DST/.lossy/summary.json`.
The values `zeros_changed` and `nan_mismatch` must be 0.

## 4. Procedure: train on a converted dataset

1. Make sure that `anemoi-lossy` is on the Python path of the training job
   (Section 2.1).
2. Make sure that the codec is found:

   ```bash
   python -c "
   import numcodecs
   c = numcodecs.get_codec({'id': 'anemoi.per_variable', 'chunk_shape': [1, 1, 1, 1], 'codecs': [[]], 'index': [0]})
   print('codec found:', type(c))"
   ```

3. Compare one sample of the copy with the source:

   ```bash
   python -c "
   from anemoi.datasets import open_dataset
   import numpy as np
   a = open_dataset('DST.zarr'); b = open_dataset('SRC.zarr')
   x, y = a[1000], b[1000]
   print(np.nanmax(np.abs(x - y)), np.array_equal(np.isnan(x), np.isnan(y)))"
   ```

4. Set the dataset path in the training configuration to the copy. For
   example, with Hydra: `system.input.dataset=DST.zarr`.
5. Do not change other items. The coordinates and statistics of the copy are
   the same as the source. Thus the graph and the normalisation do not change.

## 5. Configuration files

Each configuration has a `default` codec, a `variables` map and optional
`grid_shape` and `work_dtype`. A codec is a numcodecs configuration, or a list
of configurations that the codec applies in sequence. The keys of `variables`
are fnmatch patterns. The first pattern that matches a variable name applies.
The `bounds` section is for `check_bounds.py` only. The converter does not use
it.

| File | Dataset | Workflow | Status |
|---|---|---|---|
| `bitround-baseline.yaml` | any | bitround + zstd, numcodecs only | Pipeline test. No error bounds. |
| `o48-challenge-w{1,2,4}.yaml` | `era5-o48-2020-2021-6h-v1` | W1, W2, W4 | First tests of the challenge codecs. |
| `n320-fast.yaml` | `aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8` | fast | Used for the 25k-step training test. `w` does not meet its relative bound. |
| `n320-fast-v2.yaml` | same | fast | `w` meets its absolute and relative bounds. |
| `n320-best.yaml` | same | best | Best ratio. Too slow to decode for training. |
| `n320-single-v2-fast.yaml` | `aifs-ea-an-oper-0001-mars-n320-1979-2024-6h-v1-for-single-v2` (uuid `2a3f0534-…`) | fast | For the AIFS ens 3.0 pretraining run. NaN-aware. |
| `o2560-fast.yaml`, `o2560-fast-v2.yaml`, `o2560-best.yaml` | 4 km O2560 dataset | fast, best | Ratio tests only. |
| `challenge-sheet/*.csv` | – | – | Scoreboard tables of the challenge, for `--workflow best`. |

For `n320-single-v2-fast.yaml`, the bound types of the 149 variables are:

- 92 absolute (6 of them use bitround because of NaN);
- 16 relative (`q` levels, `hcc`, `lcc`, `mcc`);
- 17 absolute AND relative (`w` levels);
- 24 lossless.

## 6. Current results

All results are from ECMWF AG (GH200 nodes, aarch64).

### 6.1 N320 dataset (101 variables, 65,744 dates)

| Configuration | MB per date | Smaller than Blosc | Smaller than float32 | Decode, s per date, 1 core | Encode, s per date, 1 core |
|---|---|---|---|---|---|
| Source, Blosc LZ4 | 112 | 1× | 1.96× | 0.21 | – |
| `n320-fast.yaml` (full dataset) | 39.9 | 2.81× | 5.49× | 0.32 | 11 |
| `n320-fast-v2.yaml` (`w` corrected) | 49.0 | 2.29× | 4.47× | 0.3 | 13 |
| `n320-best.yaml` | 32.2 | 3.3× | 6.8× | 88 | 111 |

- The full `n320-fast.yaml` dataset is 2,442 GiB. The source is 6,859 GiB.
  The other rows are measured on 2 or 3 dates.
- The full conversion with verification took 1 h 16 min on one 288-core node.
- The winds (`u`, `v`, `10u`, `10v`, `100u`, `100v`) are 46% of the compressed
  size. Their bound (0.001 m/s) is the smallest relative to their range.

### 6.2 N320 single-v2 dataset (149 variables, 67,208 dates)

`ratio_test.py` with `n320-single-v2-fast.yaml` on 3 dates (0, 33604, 67207):

- 2.16× smaller than the source, 4.39× smaller than float32;
- 0.074 GB per date, approximately 4.5 TiB for the full dataset;
- decode 0.5 s per date on one core;
- 0 bound violations, 0 NaN mismatches.

### 6.3 O2560 dataset (4 km, 100 variables)

| Configuration | GB per date | Smaller than Blosc | Decode, s per date, 1 core | Encode, s per date, 1 core |
|---|---|---|---|---|
| Source, Blosc LZ4 | 4.70 | 1× | 8.3 | – |
| `o2560-fast.yaml` | 1.58 | 2.98× | 14 | 715 |
| `o2560-fast-v2.yaml` (`w` corrected) | 1.95 | 2.41× | 14 | 768 |
| `o2560-best.yaml` | 1.40 | 3.38× | 3,451 | 6,386 |

- The source stores one variable for each chunk. You must rechunk it to
  `[1, 100, 1, 1644160]` before conversion.
- Decode scales almost linearly with threads. With 16 threads, a date
  decodes in approximately 1 s. But each data loader worker decodes on one
  core today.

### 6.4 Training test

A GraphTransformer (anemoi-training, one GH200 GPU for each model, 25,000
steps, seed 42) was trained on the source and on the `n320-fast.yaml` copy of
the N320 dataset.

- The training loss on the lossy data was 0.3% higher.
- The validation loss (one pass, 20 batches of 2021) was 0.49% higher.
- The per-step losses of the two runs have a correlation of 0.9992.
- The training speed was 1.60 and 1.57 steps/s. The decode did not make the
  data loader slower.

Most of the loss difference comes from the quantization noise in the targets
(mainly `z` and `q`). It does not necessarily show a worse model.

## 7. Limits

- **`w` in `n320-fast.yaml`.** This configuration uses only the absolute
  bound of `w`. 48–96% of the points exceed the relative bound of 5%. The
  configurations `*-fast-v2.yaml` and `n320-single-v2-fast.yaml` correct
  this. The training test used the uncorrected data.
- **Mean bounds become pointwise bounds.** This is stricter than necessary.
  For `10u` and `10v`, the mean relative error is approximately 0.001. The
  bound is 0.5.
- **The training test is small.** It has one seed, one model, 25,000 steps
  and one validation pass. We did not calculate forecast scores.
- **The challenge codecs are too slow for training.** At 88 s per N320 date,
  training needs hundreds of data loader workers for each GPU.
- **Dataset layout.** The converter accepts only zarr v2 with one chunk that
  contains all variables of a date.
- **Configuration and dataset.** A configuration is correct only for the
  dataset that it was made from, because the offsets come from its
  statistics.

## 8. Next steps

1. **Full-size training run.** Train the AIFS ens 3.0 pretraining
   configuration (`ens_3.0_pretrain_gt+adem+land+ocean_stage1`, 32 nodes, 128
   GPUs, 300,000 steps) on the `n320-single-v2-fast.yaml` copy. Compare it
   with the reference run on the Blosc data. Keep the number of GPUs, the
   GPUs for each model and the steps the same.
2. **Forecast scores.** Score the two final checkpoints against the
   **original** data, for example with anemoi-inference on the 2023
   validation year. The loss alone cannot show model quality, because the
   targets of the lossy run contain quantization noise.
3. **More seeds.** Repeat the short training test with more seeds. Find
   whether the 0.3–0.5% loss difference is a real effect.
4. **Better codec for absolute AND relative bounds.** A codec that applies
   the smaller of the two bounds at each point can recover part of the size
   that the `w` correction costs (2.81× to 2.29×).
5. **Mean bounds.** Apply mean bounds as mean bounds, not as pointwise
   bounds. This can make the copy smaller.
6. **Parallel decode for high resolution.** At 4 km, one worker decodes a
   date in 14 s on one core. Decode the chunks of a sample in parallel (for
   example with zarr 3), or use more workers with `read_group_size` above 1.
7. **Faster challenge codecs.** Find where the decode time of the challenge
   codecs goes (context-mixing coders, wasm zstd). Native code or threaded
   decode for each variable can make them usable for training.

## 9. Repository layout

| Path | Contents |
|---|---|
| `src/anemoi_lossy/codec.py` | The `anemoi.per_variable` codec. |
| `src/anemoi_lossy/convert.py` | The converter steps `prepare`, `convert` and `finalize`. |
| `scripts/convert_dataset.sh` | Parallel converter for one machine (`xargs -P` over blocks of dates). |
| `scripts/slurm/*.sbatch` | Slurm templates for AG (multi-node). Change the paths before use. |
| `scripts/make_challenge_config.py` | Makes a configuration from the challenge requirements. |
| `scripts/ratio_test.py` | Measures ratio, speed and errors on some dates, with no output dataset. |
| `scripts/check_bounds.py` | Compares the verified errors with the bounds of the configuration. |
| `configs/` | Converter configurations (Section 5). |
| `tests/make_toy_dataset.py` | Writes a small synthetic dataset in the anemoi layout. |
| `tests/test_codec.py` | Unit tests of the codec and of the configuration patterns. |
