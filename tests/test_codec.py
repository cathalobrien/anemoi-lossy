import numpy as np
import pytest
from numcodecs.registry import get_codec

from anemoi_lossy.convert import build_codec_config

ZSTD = {"id": "zstd", "level": 1}
BITROUND = [{"id": "bitround", "keepbits": 6}, ZSTD]


@pytest.mark.parametrize("chunk_shape", [[1, 3, 1, 12], [2, 3, 4, 12]])
@pytest.mark.parametrize("grid_shape", [None, [3, 4]])
@pytest.mark.parametrize("work_dtype", [None, "float64"])
def test_roundtrip(chunk_shape, grid_shape, work_dtype):
    x = np.random.default_rng(0).normal(size=chunk_shape).astype(np.float32)
    x[0, 1, 0, 3] = np.nan
    codec = get_codec(
        {
            "id": "anemoi.per_variable",
            "chunk_shape": chunk_shape,
            "codecs": [[ZSTD], BITROUND],
            "index": [0, 1, 0],
            "grid_shape": grid_shape,
            "work_dtype": work_dtype,
        }
    )
    y = codec.decode(codec.encode(x))
    assert y.shape == x.shape and y.dtype == x.dtype
    np.testing.assert_array_equal(y[:, [0, 2]], x[:, [0, 2]])  # lossless variables
    np.testing.assert_allclose(y[:, 1], x[:, 1], rtol=2**-6)
    # flat buffers, as zarr may pass them
    out = np.empty(x.size, dtype=x.dtype)
    codec.decode(codec.encode(x.ravel()), out=out)
    np.testing.assert_array_equal(out.reshape(x.shape), y)


def test_config_patterns():
    cfg = build_codec_config(
        {"default": BITROUND, "variables": {"lsm": ZSTD, "z_*": [ZSTD]}},
        ["2t", "lsm", "z_500", "z_850", "z"],
        [1, 5, 1, 12],
    )
    assert cfg["codecs"] == [BITROUND, [ZSTD]]
    assert cfg["index"] == [0, 1, 1, 1, 0]


def test_bad_grid_shape():
    with pytest.raises(ValueError):
        get_codec({"id": "anemoi.per_variable", "chunk_shape": [1, 1, 1, 12], "codecs": [ZSTD], "index": [0], "grid_shape": [5, 5]})
