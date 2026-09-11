"""Native CPU MoE (libcpu_moe.so: AVX2/FMA + OpenMP) for DeepSeek-V4.1 MXFP4 routed experts.

Reads the HF shards directly (no requantization), runs the selected experts on the host and
returns bf16. Two entry points: ``forward_cpu`` for tests (CPU tensors) and ``forward`` for
serving (CUDA tensors: D2H into pinned staging -> cudaLaunchHostFunc -> H2D, all on the
current stream, so it sits inside CUDA graphs like the Engram SSD lookup).
"""
from __future__ import annotations

import ctypes as C
import json
import os
import struct
from pathlib import Path

import torch

_LIB = None
_P = C.c_void_p


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = C.CDLL(str(Path(__file__).with_name("libcpu_moe.so")))
        _LIB.cpu_moe_create.restype = _P
        _LIB.cpu_moe_create.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float, C.c_int]
        _LIB.cpu_moe_load.restype = C.c_int
        _LIB.cpu_moe_load.argtypes = [_P, C.c_char_p, _P]
        _LIB.cpu_moe_forward.restype = None
        _LIB.cpu_moe_forward.argtypes = [_P, C.c_int, _P, _P, _P, C.c_int, _P]
        _LIB.cpu_moe_free.argtypes = [_P]
        _LIB.cpu_moe_host_fn.restype = None
        _LIB.cpu_moe_host_fn.argtypes = [_P]
    return _LIB


class _Work(C.Structure):
    _fields_ = [("layer", _P), ("M", C.c_int), ("K", C.c_int), ("x", _P), ("ids", _P), ("wts", _P), ("out", _P)]


_CUDART = None


def _cudart():
    global _CUDART
    if _CUDART is None:
        _CUDART = C.CDLL("libcudart.so.13") if _try("libcudart.so.13") else C.CDLL("libcudart.so.12")
        _CUDART.cudaLaunchHostFunc.restype = C.c_int
        _CUDART.cudaLaunchHostFunc.argtypes = [_P, _P, _P]
    return _CUDART


def _try(name):
    try:
        C.CDLL(name); return True
    except OSError:
        return False


def expert_offsets(model_dir: str, layer: int, num_experts: int):
    """(shard path, int64 [E,6] absolute byte offsets of w1,s1,w3,s3,w2,s2) for a layer."""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    names = [f"layers.{layer}.ffn.experts.{e}.{t}" for e in range(num_experts) for t in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")]
    files = {idx[n] for n in names}
    assert len(files) == 1, f"layer {layer} experts span shards {files}"
    path = os.path.join(model_dir, files.pop())
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    offs = torch.empty(num_experts, 6, dtype=torch.int64)
    for i, n in enumerate(names):
        offs[i // 6, i % 6] = base + hdr[n]["data_offsets"][0]
    return path, offs


class NativeCpuMoe:
    def __init__(self, model_dir: str, layer: int, num_experts: int, hidden: int, inter: int,
                 swiglu_limit: float, threads: int | None = None, max_tokens: int = 2048):
        self.E, self.H, self.I = num_experts, hidden, inter
        self.threads = threads or int(os.environ.get("DSV41_CPU_EXPERT_THREADS", "16"))
        self.max_tokens = max_tokens
        self._h = _lib().cpu_moe_create(num_experts, hidden, inter, float(swiglu_limit), self.threads)
        if not self._h:
            raise RuntimeError("cpu_moe_create failed")
        self._staging: dict[tuple[int, int], tuple] = {}
        self._works: dict[tuple[int, int], _Work] = {}
        self.layer = layer
        self.model_dir = model_dir

    def load(self) -> None:
        path, offs = expert_offsets(self.model_dir, self.layer, self.E)
        offs = offs.contiguous()
        rc = _lib().cpu_moe_load(self._h, path.encode(), offs.data_ptr())
        if rc != 0:
            raise RuntimeError(f"cpu_moe_load failed rc={rc} ({path})")

    # --- CPU path (tests) ---
    def forward_cpu(self, x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        M, K = ids.shape
        x = x.to(torch.bfloat16).contiguous().cpu(); ids = ids.to(torch.int32).contiguous().cpu()
        wts = wts.to(torch.float32).contiguous().cpu()
        out = torch.empty(M, self.H, dtype=torch.bfloat16)
        _lib().cpu_moe_forward(self._h, M, x.data_ptr(), ids.data_ptr(), wts.data_ptr(), K, out.data_ptr())
        return out

    # --- CUDA-stream path (serving) ---
    def _buffers(self, device: torch.device, M: int, K: int):
        cap = 1 << max(0, (M - 1).bit_length())
        key = (device.index, cap)
        if key not in self._staging:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"cpu_moe staging {key} must be warmed up before graph capture")
            cpu = torch.device("cpu")
            x = torch.empty((cap, self.H), dtype=torch.bfloat16, device=cpu, pin_memory=True)
            ids = torch.empty((cap, K), dtype=torch.int32, device=cpu, pin_memory=True)
            wts = torch.empty((cap, K), dtype=torch.float32, device=cpu, pin_memory=True)
            out = torch.empty((cap, self.H), dtype=torch.bfloat16, device=cpu, pin_memory=True)
            dout = torch.empty((cap, self.H), dtype=torch.bfloat16, device=device)
            self._staging[key] = (x, ids, wts, out, dout)
        return self._staging[key]

    def warmup(self, device: torch.device, K: int) -> None:
        c = 1
        while c <= self.max_tokens * 2:
            self._buffers(device, c, K); c *= 2

    def forward(self, x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        M, K = ids.shape
        device = x.device
        hx, hids, hw, hout, dout = self._buffers(device, M, K)
        hx[:M].copy_(x.to(torch.bfloat16), non_blocking=True)
        hids[:M].copy_(ids.to(torch.int32), non_blocking=True)
        hw[:M].copy_(wts.to(torch.float32), non_blocking=True)
        key = (device.index, M)
        if key not in self._works:
            self._works[key] = _Work(self._h, M, K, hx.data_ptr(), hids.data_ptr(), hw.data_ptr(), hout.data_ptr())
        work = self._works[key]
        stream = torch.cuda.current_stream(device).cuda_stream
        err = _cudart().cudaLaunchHostFunc(stream, C.cast(_lib().cpu_moe_host_fn, _P), C.addressof(work))
        if err:
            raise RuntimeError(f"cudaLaunchHostFunc failed: {err}")
        dout[:M].copy_(hout[:M], non_blocking=True)
        return dout[:M]

    def close(self):
        if self._h:
            _lib().cpu_moe_free(self._h); self._h = None
