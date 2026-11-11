"""Per-variable numcodecs wrapper for anemoi dataset chunks.

An anemoi ``data`` array has shape ``(dates, variables, ensembles, values)``.
A zarr v2 array has one compressor, so this codec divides each chunk along the
variable axis and applies a separate codec chain to each variable.

The chunks can contain all variables (``(1, n_variables, 1, n_values)``, the
anemoi-datasets default), or only some variables and some grid points (for
example ``(1, 1, 1, n_values / 16)``). Zarr does not tell a codec the position
of a chunk. Thus, for chunks that do not contain all variables, the encoder
must get the index of the first variable of the chunk from
:meth:`PerVariable.encode_block`. The encoder writes the chain of each
variable into the chunk header, and the decoder reads it from there. Zarr can
read these arrays, but it cannot write them (``encode`` raises an error).

Config (as stored in ``data/.zarray`` -> ``compressor``)::

    {
      "id": "anemoi.per_variable",
      "chunk_shape": [1, 100, 1, 40320],   # chunk shape of the array
      "codecs": [[{...}, {...}], [{...}]], # unique codec chains
      "index": [0, 0, 1, ...],             # dataset variable i -> codecs[index[i]]
      "grid_shape": [721, 1440] | null,    # reshape values axis before encoding
      "work_dtype": "float64" | null       # cast slices before encoding
    }

Encoded layout::

    b"APV1" | u32 header_len | header (JSON: dtype, shape, lengths, chains) | payloads

``chains`` gives the index into ``codecs`` for each variable of the chunk. An
edge chunk can extend past the last variable of the array. Zarr ignores these
padding variables, so the encoder does not store them, and ``chains`` is then
shorter than the chunk.
Chunks written before this key was added do not have it. For these chunks,
the decoder uses ``index`` (they always contain all variables).
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
        if self.chunk_shape[1] > len(self.index):
            raise ValueError(f"index has {len(self.index)} entries but the chunk has {self.chunk_shape[1]} variables")
        if any(not 0 <= i < len(self.codecs) for i in self.index):
            raise ValueError(f"index has entries outside 0..{len(self.codecs) - 1}")
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
        """Encode a chunk that contains all variables of the dataset."""
        if self.chunk_shape[1] != len(self.index):
            raise ValueError(
                "the chunks of this array do not contain all variables: zarr cannot write them. "
                "Use PerVariable.encode_block (anemoi_lossy.convert does this)."
            )
        return self.encode_block(buf, 0)

    def encode_block(self, buf, first_variable):
        """Encode a chunk whose first variable has the index ``first_variable``
        in the dataset."""
        arr = ensure_ndarray(buf)
        if arr.ndim != 4:
            arr = arr.reshape(self.chunk_shape)
        first_variable = int(first_variable)
        if not 0 <= first_variable < len(self.index):
            raise ValueError(f"variable {first_variable} is not in the index of {len(self.index)} variables")
        # an edge chunk can extend past the last variable: zarr ignores that padding, so it is not stored
        chains = self.index[first_variable : first_variable + arr.shape[1]]
        dtype = arr.dtype
        work = np.dtype(self.work_dtype) if self.work_dtype else dtype
        slice_shape = self._slice_shape(arr.shape)

        payloads = []
        for i in range(len(chains)):
            x = np.ascontiguousarray(arr[:, i]).reshape(slice_shape).astype(work, copy=False)
            for codec in self._chains[chains[i]]:
                # some codecs (e.g. the wasm ones) reject bytes objects: pass arrays between steps
                x = ensure_ndarray(codec.encode(x))
            payloads.append(ensure_bytes(x))

        header = json.dumps(
            {"dtype": dtype.str, "shape": list(arr.shape), "lengths": [len(p) for p in payloads], "chains": chains}
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
        chains = header.get("chains", self.index[:nvar])
        if len(chains) != len(header["lengths"]) or len(chains) > nvar:
            raise ValueError(f"chunk header has {len(chains)} chains for {len(header['lengths'])} payloads")

        result = np.empty(shape, dtype=dtype)
        result[:, len(chains) :] = 0  # padding of an edge chunk (not stored)
        offset = 8 + hlen
        for i, length in enumerate(header["lengths"]):
            x = ensure_ndarray(data[offset : offset + length])
            offset += length
            for codec in reversed(self._chains[chains[i]]):
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
