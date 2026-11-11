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


def test_encode_block_split_variables():
    # chunks with one variable: the chain of each chunk comes from the header
    codec = get_codec(
        {"id": "anemoi.per_variable", "chunk_shape": [1, 1, 1, 12], "codecs": [[ZSTD], BITROUND], "index": [0, 1, 0]}
    )
    x = np.random.default_rng(0).normal(size=(1, 1, 1, 12)).astype(np.float32)
    np.testing.assert_array_equal(codec.decode(codec.encode_block(x, 0)), x)  # lossless chain
    y = codec.decode(codec.encode_block(x, 1))  # bitround chain
    assert not np.array_equal(y, x)
    np.testing.assert_allclose(y, x, rtol=2**-6)
    with pytest.raises(ValueError, match="encode_block"):
        codec.encode(x)  # zarr does not know the variable of the chunk
    with pytest.raises(ValueError):
        codec.encode_block(x, 3)


def test_decode_chunk_without_chains():
    # chunks written before the 'chains' header key was added must still decode
    import json
    import struct

    codec = get_codec({"id": "anemoi.per_variable", "chunk_shape": [1, 2, 1, 12], "codecs": [[ZSTD]], "index": [0, 0]})
    x = np.random.default_rng(0).normal(size=(1, 2, 1, 12)).astype(np.float32)
    enc = codec.encode(x)
    (hlen,) = struct.unpack("<I", enc[4:8])
    header = json.loads(enc[8 : 8 + hlen])
    del header["chains"]
    h = json.dumps(header).encode()
    old = enc[:4] + struct.pack("<I", len(h)) + h + enc[8 + hlen :]
    np.testing.assert_array_equal(codec.decode(old), x)
