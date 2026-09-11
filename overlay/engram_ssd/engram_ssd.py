# SPDX-License-Identifier: Apache-2.0
"""Engram tables served from the checkpoint shards on the NVMe.

Replaces the pinned-host (UVA) storage of vLLM's ParallelEngramEmbedding when
DSV41_ENGRAM_STORAGE=ssd. The n-gram hash ids computed on the GPU are copied to a
pinned host buffer, a host callback (cudaLaunchHostFunc, so it is legal inside
CUDA graphs) preads the FP8 rows + UE8M0 scales straight from the safetensors
shard through the kernel page cache (librow_store.so, thread pool), the rows are
copied back to the device and dequantized there with plain torch ops (no Triton
fp8 converts, which sm_80 lacks).

Nothing about the model changes: hashing, gating, wkv projection and the
tensor-parallel head sharding stay exactly as in the reference implementation.
"""
from __future__ import annotations

import ctypes as C
import glob
import json
import os
import struct
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_P, _U = C.c_void_p, C.c_uint64
_LIB: C.CDLL | None = None
_CUDART: C.CDLL | None = None


class _Work(C.Structure):
    _fields_ = [
        ("store", _P),
        ("ids", _P),
        ("weights", _P),
        ("scales", _P),
        ("count", _U),
    ]


def _lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        so = os.environ.get(
            "DSV41_ROW_STORE_SO",
            str(Path(__file__).with_name("librow_store.so")),
        )
        _LIB = C.CDLL(so)
        _LIB.row_store_open.argtypes = [C.c_char_p, _U, _U, _U, C.c_int]
        _LIB.row_store_open.restype = _P
        _LIB.row_store_range.argtypes = [_P, _U, _U]
        _LIB.row_store_lookup_sync.argtypes = [_P, _P, _P, _P, _U]
        _LIB.row_store_stats.argtypes = [_P, _P]
        _LIB.row_store_close.argtypes = [_P]
    return _LIB


def _cudart() -> C.CDLL:
    global _CUDART
    if _CUDART is None:
        cands = glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so*")
        cands += glob.glob(
            "/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*"
        )
        cands += glob.glob(
            "/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"
        )
        cands += glob.glob(
            os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")
        )
        if not cands:
            raise RuntimeError("libcudart.so not found for cudaLaunchHostFunc")
        _CUDART = C.CDLL(cands[0])
        _CUDART.cudaLaunchHostFunc.argtypes = [_P, _P, _P]
        _CUDART.cudaLaunchHostFunc.restype = C.c_int
    return _CUDART


def find_engram_shard(model_dir: str, layer_id: int) -> tuple[str, int, int, int]:
    """(shard path, rows, weight byte offset, scale byte offset) of one table."""
    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer_id}.engram.embed."
    wkey, skey = prefix + "weight", prefix + "scale"
    if index[wkey] != index[skey]:
        raise RuntimeError("engram weight and scale live in different shards")
    shard = root / index[wkey]
    with shard.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    w, s = header[wkey], header[skey]
    rows = w["shape"][0]
    if w["dtype"] != "F8_E4M3" or w["shape"] != [rows, 256]:
        raise RuntimeError(f"unexpected engram weight {w}")
    if s["dtype"] != "F8_E8M0" or s["shape"] != [rows, 8]:
        raise RuntimeError(f"unexpected engram scale {s}")
    base = 8 + n
    return str(shard), rows, base + w["data_offsets"][0], base + s["data_offsets"][0]


class SsdEngramTable:
    """Row provider for one Engram layer (one TP rank's head range).

    lookup(indices, out): indices [T, n_hash_cols] int32/int64 on the device,
    out [T, local_heads, 256] bf16 on the device; only heads
    [head_start, head_start + local_heads) are read, ids outside
    [vocab_start, vocab_end) yield zero rows (matches the UVA kernel).
    """

    def __init__(
        self,
        model_dir: str,
        layer_id: int,
        vocab_start: int,
        vocab_end: int,
        head_start: int,
        local_heads: int,
        total_heads: int,
        dim: int = 256,
        block_size: int = 32,
        max_tokens: int = 8192,
        nthreads: int | None = None,
    ) -> None:
        assert dim == 256 and block_size == 32
        self.dim, self.block_size = dim, block_size
        self.head_start, self.local_heads, self.total_heads = (
            head_start,
            local_heads,
            total_heads,
        )
        self.vocab_start, self.vocab_end = vocab_start, vocab_end
        path, rows, woff, soff = find_engram_shard(model_dir, layer_id)
        self.rows = rows
        if nthreads is None:
            nthreads = int(os.environ.get("DSV41_ENGRAM_THREADS", "16"))
        self._store = _lib().row_store_open(path.encode(), rows, woff, soff, nthreads)
        if not self._store:
            raise RuntimeError(f"could not open engram shard {path}")
        _lib().row_store_range(self._store, vocab_start, vocab_end)
        self._staging: dict[tuple[int, int], tuple] = {}
        self._works: dict[tuple[int, int], _Work] = {}
        self.max_tokens = max_tokens
        self._exp_shift = 23
        if torch.cuda.is_available():
            self.warmup(torch.device("cuda", torch.cuda.current_device()))
        logger.info(
            "Engram layer %d served from SSD: %s rows [%d, %d) heads [%d, %d) "
            "threads=%d",
            layer_id,
            path,
            vocab_start,
            vocab_end,
            head_start,
            head_start + local_heads,
            nthreads,
        )

    # --- staging -----------------------------------------------------------
    def _buffers(self, device: torch.device, count: int):
        capacity = 1 << max(0, (count - 1).bit_length())
        key = (device.index, capacity)
        if key not in self._staging:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"engram staging {key} must be warmed up before graph capture"
                )
            # Explicit devices: model init runs under a torch.device("cuda") context.
            cpu = torch.device("cpu")
            ids = torch.empty(capacity, dtype=torch.int64, device=cpu, pin_memory=True)
            w = torch.empty((capacity, self.dim), dtype=torch.uint8, device=cpu, pin_memory=True)
            s = torch.empty(
                (capacity, self.dim // self.block_size), dtype=torch.uint8, device=cpu, pin_memory=True
            )
            dw = torch.empty((capacity, self.dim), dtype=torch.uint8, device=device)
            ds = torch.empty((capacity, self.dim // self.block_size), dtype=torch.uint8, device=device)
            self._staging[key] = (ids, w, s, dw, ds)
        return self._staging[key]

    def warmup(self, device: torch.device, max_tokens: int | None = None) -> None:
        """Allocate every staging capacity (powers of two up to max_tokens x
        local_heads) before any CUDA graph capture."""
        max_rows = (max_tokens or self.max_tokens) * self.local_heads
        c = 1
        while c <= max_rows * 2:
            self._buffers(device, c)
            c *= 2

    # --- lookup ------------------------------------------------------------
    def lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        num_tokens = indices.shape[0]
        count = num_tokens * self.local_heads
        if count == 0:
            return
        if os.environ.get("DSV41_ENGRAM_ZERO") == "1":  # diagnostic: no memory rows
            out.zero_()
            return
        device = indices.device
        ids, w, s, dw, ds = self._buffers(device, count)
        work_key = (device.index, count)
        if work_key not in self._works:
            self._works[work_key] = _Work(
                self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count
            )
        work = self._works[work_key]
        local = indices[:, self.head_start : self.head_start + self.local_heads]
        # D2H of the ids on the current stream (pinned target => async).
        ids[:count].copy_(local.reshape(-1).to(torch.int64), non_blocking=True)
        stream = torch.cuda.current_stream(device).cuda_stream
        err = _cudart().cudaLaunchHostFunc(
            stream, C.cast(_lib().row_store_lookup, _P), C.addressof(work)
        )
        if err:
            raise RuntimeError(f"cudaLaunchHostFunc failed: {err}")
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
        # Dequant on the device: fp8 e4m3 * 2^(e8m0 - 127), per 32 columns.
        vals = dw[:count].view(torch.float8_e4m3fn).to(torch.float32)
        scale = (ds[:count].to(torch.int32) << self._exp_shift).view(torch.float32)
        scale = scale.repeat_interleave(self.block_size, dim=1)
        out.view(-1, self.dim)[:count].copy_((vals * scale).to(torch.bfloat16))

    def stats(self) -> tuple[int, int]:
        buf = (C.c_uint64 * 2)()
        _lib().row_store_stats(self._store, C.addressof(buf))
        return int(buf[0]), int(buf[1])

    def close(self) -> None:
        if self._store:
            _lib().row_store_close(self._store)
            self._store = None


def reference_lookup_cpu(
    model_dir: str, layer_id: int, ids: torch.Tensor
) -> torch.Tensor:
    """Slow, exact CPU reference used by tests: ids [N] -> bf16 [N, 256]."""
    import numpy as np

    path, rows, woff, soff = find_engram_shard(model_dir, layer_id)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    idn = ids.cpu().numpy().astype(np.int64)
    w = np.stack([mm[woff + i * 256 : woff + (i + 1) * 256] for i in idn])
    s = np.stack([mm[soff + i * 8 : soff + (i + 1) * 8] for i in idn])
    vals = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).to(torch.float32)
    scale = (torch.from_numpy(s.copy()).to(torch.int32) << 23).view(torch.float32)
    return (vals * scale.repeat_interleave(32, dim=1)).to(torch.bfloat16)
