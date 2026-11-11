"""End-to-end tests of the converter for different chunk layouts."""

import sys
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).parent))
from make_toy_dataset import VARIABLES, make  # noqa: E402

from anemoi_lossy import convert  # noqa: E402

KEEPBITS = 8
CONFIG = {
    "default": [{"id": "bitround", "keepbits": KEEPBITS}, {"id": "zstd", "level": 1}],
    "variables": {"lsm": {"id": "zstd", "level": 1}, "cos_*": {"id": "zstd", "level": 1}},
}
LOSSLESS = [VARIABLES.index("lsm"), VARIABLES.index("cos_latitude")]


@pytest.mark.parametrize(
    "var_chunk, value_chunk",
    [
        (None, None),  # all variables and grid points in one chunk (anemoi-datasets default)
        (1, None),  # one variable per chunk
        (1, 4095),  # one variable per chunk, grid in 4 equal blocks
        (3, 5000),  # 3 variables per chunk, grid in blocks; edge chunks on both axes
    ],
)
def test_convert_layouts(tmp_path, var_chunk, value_chunk):
    src, dst, cfg = tmp_path / "src.zarr", tmp_path / "dst.zarr", tmp_path / "config.yaml"
    n_dates = 3
    data = make(str(src), n_dates, var_chunk, value_chunk)
    cfg.write_text(yaml.safe_dump(CONFIG))

    convert.prepare(src, dst, cfg)
    convert.convert(src, dst, 0, n_dates, verify=True, overwrite=False)
    convert.finalize(src, dst)

    out = zarr.open_group(str(dst), mode="r")["data"][:]
    assert out.shape == data.shape and out.dtype == data.dtype
    np.testing.assert_array_equal(np.isnan(out), np.isnan(data))
    np.testing.assert_array_equal(out[:, LOSSLESS], data[:, LOSSLESS])
    np.testing.assert_allclose(out, data, rtol=2.0 ** -(KEEPBITS + 1), equal_nan=True)

    summary = yaml.safe_load((dst / ".lossy" / "summary.json").read_text())
    for i, name in enumerate(VARIABLES):
        v = summary["variables"][name]
        assert v["nan_mismatch"] == 0
        if i in LOSSLESS:
            assert v["max_abs_error"] == 0
        else:
            assert 0 < v["max_rel_error"] <= 2.0 ** -(KEEPBITS + 1)


def test_zarr_cannot_write_split_variables(tmp_path):
    src, dst, cfg = tmp_path / "src.zarr", tmp_path / "dst.zarr", tmp_path / "config.yaml"
    make(str(src), 1, 1, None)
    cfg.write_text(yaml.safe_dump(CONFIG))
    convert.prepare(src, dst, cfg)
    arr = zarr.open_group(str(dst), mode="r+")["data"]
    with pytest.raises(ValueError, match="encode_block"):
        arr[0] = 0.0
