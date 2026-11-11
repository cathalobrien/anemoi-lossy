"""Write a small synthetic dataset with the anemoi-datasets layout (zarr v2,
data (dates, variables, 1, values), Blosc) for testing.

    python tests/make_toy_dataset.py OUT.zarr [n_dates [variables_per_chunk [values_per_chunk]]]

By default, each chunk contains all variables and grid points of one date.
Give variables_per_chunk and values_per_chunk to divide each date into more
chunks.
"""

import sys

import numcodecs
import numpy as np
import zarr

VARIABLES = ["2t", "10u", "msl", "sst", "t_500", "z_500", "lsm", "cos_latitude"]


def make(out, n_dates=24, var_chunk=None, value_chunk=None):
    """Write the dataset to OUT. Return the array of the data."""
    ny, nx = 91, 180  # 2 deg regular grid
    variables = VARIABLES
    var_chunk = var_chunk or len(variables)
    value_chunk = value_chunk or ny * nx
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
        chunks=(1, var_chunk, 1, value_chunk),
        dtype="float32",
        compressors=numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE),
        fill_value=np.nan,
    )
    arr[:] = data
    t0 = np.datetime64("2020-01-01T00")
    dates = np.arange(t0, t0 + np.timedelta64(n_dates * 6, "h"), np.timedelta64(6, "h"))
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
    print(f"wrote {out}: data {data.shape}, chunks {arr.chunks}")
    return data


if __name__ == "__main__":
    a = sys.argv[1:]
    make(a[0], *(int(x) for x in a[1:]))
