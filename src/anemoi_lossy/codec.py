"""Per-variable numcodecs wrapper for anemoi dataset chunks.

An anemoi ``data`` array has shape ``(dates, variables, ensembles, values)``
and is normally chunked ``(1, n_variables, 1, n_values)``. A zarr v2 array has
one compressor, so this codec splits each chunk along the variable axis and
applies a separate codec chain to every variable.

Config (as stored in ``data/.zarray`` -> ``compressor``)::

    {
      "id": "anemoi.per_variable",
      "chunk_shape": [1, 100, 1, 40320],   # full chunk shape of the array
      "codecs": [[{...}, {...}], [{...}]], # unique codec chains
      "index": [0, 0, 1, ...],             # variable i -> codecs[index[i]]
      "grid_shape": [721, 1440] | null,    # reshape values axis before encoding
      "work_dtype": "float64" | null       # cast slices before encoding
    }

Encoded layout::

    b"APV1" | u32 header_len | header (JSON: dtype, shape, lengths) | payloads
"""

from __future__ import annotations

import json
import struct

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_bytes
from numcodecs.compat import ensure_ndarray
from numcodecs.registry import get_codec

MAGIC = b"APV1"


def _as_chain(cfg: dict | list) -> list[dict]:
    return list(cfg) if isinstance(cfg, (list, tuple)) else [cfg]


class PerVariable(Codec):
    codec_id = "anemoi.per_variable"

    def __init__(self, chunk_shape, codecs, index, grid_shape=None, work_dtype=None):
        self.chunk_shape = [int(s) for s in chunk_shape]
        self.codecs = [_as_chain(c) for c in codecs]
        self.index = [int(i) for i in index]
        self.grid_shape = None if grid_shape is None else [int(s) for s in grid_shape]
        self.work_dtype = work_dtype

        if len(self.chunk_shape) != 4:
            raise ValueError(f"chunk_shape must be (dates, variables, ensembles, values), got {self.chunk_shape}")
        if len(self.index) != self.chunk_shape[1]:
            raise ValueError(f"index has {len(self.index)} entries but the chunk has {self.chunk_shape[1]} variables")
        if self.grid_shape is not None and int(np.prod(self.grid_shape)) != self.chunk_shape[3]:
            raise ValueError(f"grid_shape {self.grid_shape} does not match {self.chunk_shape[3]} values")

        self._chains = [[get_codec(c) for c in chain] for chain in self.codecs]

    def get_config(self):
        return {
            "id": self.codec_id,
            "chunk_shape": self.chunk_shape,
            "codecs": self.codecs,
            "index": self.index,
            "grid_shape": self.grid_shape,
            "work_dtype": self.work_dtype,
        }

    def _slice_shape(self, shape):
        t, _, e, g = shape
        lead = [t * e] if t * e > 1 else []
        return lead + (self.grid_shape if self.grid_shape is not None else [g])

    def encode(self, buf):
        arr = ensure_ndarray(buf)
        if arr.ndim != 4:
            arr = arr.reshape(self.chunk_shape)
        dtype = arr.dtype
        work = np.dtype(self.work_dtype) if self.work_dtype else dtype
        slice_shape = self._slice_shape(arr.shape)

        payloads = []
        for i in range(arr.shape[1]):
            x = np.ascontiguousarray(arr[:, i]).reshape(slice_shape).astype(work, copy=False)
            for codec in self._chains[self.index[i]]:
                # some codecs (e.g. the wasm ones) reject bytes objects: pass arrays between steps
                x = ensure_ndarray(codec.encode(x))
            payloads.append(ensure_bytes(x))

        header = json.dumps(
            {"dtype": dtype.str, "shape": list(arr.shape), "lengths": [len(p) for p in payloads]}
        ).encode()
        return MAGIC + struct.pack("<I", len(header)) + header + b"".join(payloads)

    def decode(self, buf, out=None):
        data = memoryview(ensure_bytes(buf))
        if bytes(data[:4]) != MAGIC:
            raise ValueError("not an anemoi.per_variable chunk")
        (hlen,) = struct.unpack("<I", data[4:8])
        header = json.loads(bytes(data[8 : 8 + hlen]))
        dtype = np.dtype(header["dtype"])
        shape = header["shape"]
        work = np.dtype(self.work_dtype) if self.work_dtype else dtype
        t, nvar, e, g = shape

        result = np.empty(shape, dtype=dtype)
        offset = 8 + hlen
        for i, length in enumerate(header["lengths"]):
            x = ensure_ndarray(data[offset : offset + length])
            offset += length
            for codec in reversed(self._chains[self.index[i]]):
                x = ensure_ndarray(codec.decode(x))
            x = ensure_ndarray(x)
            # chains ending in a byte codec return raw bytes; others return the array
            x = x.astype(dtype, copy=False) if x.dtype.kind == "f" else x.view(work).astype(dtype, copy=False)
            result[:, i] = x.reshape(t, e, g)

        if out is not None:
            out = ensure_ndarray(out).view(dtype)
            out[...] = result.reshape(out.shape)
            return out
        return result
