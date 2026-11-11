"""Write a small synthetic dataset with the anemoi-datasets layout (zarr v2,
data (dates, variables, 1, values) chunked per date, Blosc) for testing.

    python tests/make_toy_dataset.py OUT.zarr [n_dates]
"""

import sys

import numcodecs
import numpy as np
import zarr

out = sys.argv[1]
n_dates = int(sys.argv[2]) if len(sys.argv) > 2 else 24
ny, nx = 91, 180  # 2 deg regular grid
variables = ["2t", "10u", "msl", "sst", "t_500", "z_500", "lsm", "cos_latitude"]

lat = np.linspace(90, -90, ny)
lon = np.linspace(0, 360, nx, endpoint=False)
LON, LAT = np.meshgrid(lon, lat)
rng = np.random.default_rng(0)

data = np.empty((n_dates, len(variables), 1, ny * nx), dtype=np.float32)
for t in range(n_dates):
    wave = np.sin(np.radians(LON * 2 + t * 15)) * np.cos(np.radians(LAT))
    fields = {
        "2t": 288 - 40 * np.abs(LAT) / 90 + 5 * wave + rng.normal(0, 0.5, LAT.shape),
        "10u": 8 * wave + rng.normal(0, 1, LAT.shape),
        "msl": 101325 + 1500 * wave,
        "sst": np.where(np.abs(LAT) < 60, 290 - 20 * np.abs(LAT) / 60 + wave, np.nan),
        "t_500": 250 - 20 * np.abs(LAT) / 90 + 3 * wave,
        "z_500": 55000 + 3000 * np.cos(np.radians(LAT)) + 500 * wave,
        "lsm": (np.abs(LAT) >= 60).astype(float),
        "cos_latitude": np.cos(np.radians(LAT)),
    }
    for i, v in enumerate(variables):
        data[t, i, 0] = fields[v].ravel()

root = zarr.open_group(out, mode="w", zarr_format=2)
arr = root.create_array(
    "data",
    shape=data.shape,
    chunks=(1, len(variables), 1, ny * nx),
    dtype="float32",
    compressors=numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE),
    fill_value=np.nan,
)
arr[:] = data
dates = np.arange(np.datetime64("2020-01-01T00"), np.datetime64("2020-01-01T00") + n_dates * 6, 6)
root.create_array("dates", data=dates.astype("datetime64[s]"))
root.create_array("latitudes", data=LAT.ravel())
root.create_array("longitudes", data=LON.ravel())
for name, fn in [("mean", np.nanmean), ("stdev", np.nanstd), ("minimum", np.nanmin), ("maximum", np.nanmax)]:
    root.create_array(name, data=fn(data, axis=(0, 2, 3)).astype(np.float64))
root.attrs.update(
    {
        "variables": variables,
        "name_to_index": {v: i for i, v in enumerate(variables)},
        "frequency": "6h",
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "resolution": "2",
        "field_shape": [ny, nx],
        "missing_dates": [],
    }
)
zarr.consolidate_metadata(out)
print(f"wrote {out}: data {data.shape}")
