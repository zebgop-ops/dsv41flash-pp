#!/usr/bin/env python3
"""Env-gated per-layer activation statistics (DSV41_DEBUG_STATS=1) in the V4.1 model
forward, to localize numerical breakage across ranks/layers. usage: <model.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "_dsv41_dbg_stats" in src:
    print("already patched"); sys.exit(0)
old = '''            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
'''
new = '''            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
            if _dsv41_dbg_dump_dir() and 1 < positions.shape[0] <= 64 and is_forward_context_available() and isinstance(get_forward_context().attn_metadata, dict):
                _d = _dsv41_dbg_dump_dir()
                _dsv41_os.makedirs(_d, exist_ok=True)
                _tag = f"L{idx:02d}_r{get_pp_group().rank_in_group}"
                if not _dsv41_os.path.exists(f"{_d}/{_tag}.pt"):
                    torch.save(
                        {"hidden_states": hidden_states.detach().cpu(),
                         "residual": residual.detach().cpu() if residual is not None else None,
                         "pre_mix": pre_mix.detach().cpu() if pre_mix is not None else None,
                         "post_mix": post_mix.detach().cpu() if post_mix is not None else None,
                         "res_mix": res_mix.detach().cpu() if res_mix is not None else None,
                         "positions": positions.detach().cpu(),
                         "input_ids": input_ids.detach().cpu() if input_ids is not None else None},
                        f"{_d}/{_tag}.pt")
            if _dsv41_gdump_dir():
                _dsv41_gdump_layer(self, idx, hidden_states)
            if _dsv41_dbg_timing():
                torch.cuda.synchronize()
                _now = _dsv41_time.perf_counter()
                if idx == self.start_layer:
                    self._dsv41_t_prev = _now; self._dsv41_t_first = _now
                    self._dsv41_tick = getattr(self, "_dsv41_tick", 0) + 1
                if hidden_states.shape[0] == 1 and self._dsv41_tick % 16 == 0:
                    logger.info("TIMING L%d T=%d layer %.2f ms cum %.2f ms", idx, hidden_states.shape[0],
                                (_now - self._dsv41_t_prev) * 1e3, (_now - self._dsv41_t_first) * 1e3)
                self._dsv41_t_prev = _now
            if _dsv41_dbg_stats():
                _h = hidden_states.float()
                _r = residual.float() if residual is not None else _h
                logger.info(
                    "DBG L%d T=%d h mean|.|=%.4f max=%.2f nan=%d | res mean|.|=%.4f max=%.2f",
                    idx, _h.shape[0], _h.abs().mean().item(), _h.abs().max().item(),
                    int(torch.isnan(_h).sum().item()), _r.abs().mean().item(), _r.abs().max().item(),
                )
'''
assert old in src, "loop anchor missing"
src = src.replace(old, new, 1)
# hooks on attn / ffn modules for short real batches
old3 = '''        # The n-gram hash needs a slot-keyed rolling store of compressed ids
'''
new3 = '''        if _dsv41_dbg_dump_dir():
            _dsv41_install_dump_hooks(self)

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
'''
assert old3 in src, "hook anchor missing"
src = src.replace(old3, new3, 1)
old2 = "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n"
new2 = old2 + '''import os as _dsv41_os


def _dsv41_dbg_stats() -> bool:
    return _dsv41_os.environ.get("DSV41_DEBUG_STATS") == "1"


import time as _dsv41_time


def _dsv41_dbg_timing() -> bool:
    return _dsv41_os.environ.get("DSV41_DEBUG_TIMING") == "1"


def _dsv41_gdump_dir() -> str:
    return _dsv41_os.environ.get("DSV41_DEBUG_GDUMP", "")


import ctypes as _dsv41_C

_DSV41_GD_CB = _dsv41_C.CFUNCTYPE(None, _dsv41_C.c_void_p)
_dsv41_gd_cudart = None


_DSV41_GD_STATE: dict = {}


def _dsv41_gdump_layer(model, idx, hidden_states):
    _dsv41_gdump_tensor(f"L{idx:02d}", hidden_states)


def _dsv41_gdump_tensor(key, hidden_states):
    """Graph-safe per-layer dump: D2H copy into a pinned buffer plus a cudaLaunchHostFunc
    callback that writes it to disk, so it also fires on CUDA-graph replay (the ordinary
    forward hooks do not). Files: <dir>/G<call>_L<idx>_r<rank>.npy (bf16 bits as int16)."""
    global _dsv41_gd_cudart
    import numpy as _np

    T = hidden_states.shape[0]
    if T > 64:
        return
    st = _DSV41_GD_STATE
    idx = key
    if idx not in st:
        if torch.cuda.is_current_stream_capturing():
            return  # buffers must exist before capture (warmup allocates them)
        buf = torch.empty((64, hidden_states[0].numel()), dtype=hidden_states.dtype, device="cpu", pin_memory=True)
        rank = get_pp_group().rank_in_group
        d = _dsv41_gdump_dir()
        counter = [0]
        shape = tuple(hidden_states.shape[1:])

        def cb(_):
            n = counter[0]
            counter[0] += 1
            arr = buf.view(torch.int16).numpy() if buf.dtype == torch.bfloat16 else buf.numpy()
            _np.save(f"{d}/G{n:03d}_{idx}_r{rank}.npy", arr.copy())

        st[idx] = (buf, _DSV41_GD_CB(cb), shape)
        _dsv41_os.makedirs(d, exist_ok=True)
    buf, cbp, _ = st[idx]
    buf[:T].copy_(hidden_states.reshape(T, -1), non_blocking=True)
    if T < 64:
        buf[T:].zero_()
    if _dsv41_gd_cudart is None:
        try:
            _dsv41_gd_cudart = _dsv41_C.CDLL("libcudart.so.13")
        except OSError:
            _dsv41_gd_cudart = _dsv41_C.CDLL("libcudart.so.12")
        _dsv41_gd_cudart.cudaLaunchHostFunc.restype = _dsv41_C.c_int
        _dsv41_gd_cudart.cudaLaunchHostFunc.argtypes = [_dsv41_C.c_void_p, _dsv41_C.c_void_p, _dsv41_C.c_void_p]
    stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
    _dsv41_gd_cudart.cudaLaunchHostFunc(stream, _dsv41_C.cast(cbp, _dsv41_C.c_void_p), None)


def _dsv41_dbg_dump_dir() -> str:
    return _dsv41_os.environ.get("DSV41_DEBUG_DUMP", "")


def _dsv41_install_dump_hooks(model) -> None:
    d = _dsv41_dbg_dump_dir()
    rank = get_pp_group().rank_in_group

    def _ok(x):
        return (
            1 < x.shape[0] <= 64
            and is_forward_context_available()
            and isinstance(get_forward_context().attn_metadata, dict)
        )

    def _mk(idx, kind):
        def hook(mod, args, out):
            x = args[1] if kind == "attn" else args[0]
            if not torch.is_tensor(x) or not _ok(x):
                return
            f = f"{d}/L{idx:02d}_r{rank}_{kind}.pt"
            if _dsv41_os.path.exists(f):
                return
            _dsv41_os.makedirs(d, exist_ok=True)
            rec = {"x": x.detach().cpu(), "out": out.detach().cpu()}
            if kind == "attn":
                rec["positions"] = args[0].detach().cpu()
            else:
                rec["input_ids"] = args[1].detach().cpu() if len(args) > 1 and torch.is_tensor(args[1]) else None
            torch.save(rec, f)
        return hook

    for idx, layer in enumerate(model.layers):
        if not hasattr(layer, "attn"):
            continue
        layer.attn.register_forward_hook(_mk(idx, "attn"))
        layer.ffn.register_forward_hook(_mk(idx, "ffn"))
'''
assert old2 in src; src = src.replace(old2, new2, 1)
old_eg = '''                residual = self.engram(
                    residual,
                    engram_hashes[:, self.engram.layer_hash_index],
                    engram_mask,
                )'''
new_eg = '''                if _dsv41_gdump_dir():
                    _dsv41_gdump_tensor(f"EG{self.engram.layer_hash_index}hash", engram_hashes[:, self.engram.layer_hash_index].contiguous())
                    _dsv41_gdump_tensor(f"EG{self.engram.layer_hash_index}in", residual)
                residual = self.engram(
                    residual,
                    engram_hashes[:, self.engram.layer_hash_index],
                    engram_mask,
                )
                if _dsv41_gdump_dir():
                    _dsv41_gdump_tensor(f"EG{self.engram.layer_hash_index}out", residual)'''
assert src.count(old_eg) == 1, src.count(old_eg)
src = src.replace(old_eg, new_eg, 1)

# sub-block graph dumps (layers 0-2): attention in/out and ffn in/out
old_loop = '''            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,'''
new_loop = '''            if _dsv41_gdump_dir():
                layer._dsv41_idx = idx
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,'''
assert src.count(old_loop) == 1, src.count(old_loop)
src = src.replace(old_loop, new_loop, 1)
old_attn = '''        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)'''
new_attn = '''        _gi = getattr(self, "_dsv41_idx", -1)
        if _dsv41_gdump_dir() and 0 <= _gi < 3:
            _dsv41_gdump_tensor(f"A{_gi:02d}in", x)
        x = self.attn(positions, x, None)
        if _dsv41_gdump_dir() and 0 <= _gi < 3:
            _dsv41_gdump_tensor(f"A{_gi:02d}out", x)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)'''
assert src.count(old_attn) == 1, src.count(old_attn)
src = src.replace(old_attn, new_attn, 1)
old_ffn = '''        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre'''
new_ffn = '''        if _dsv41_gdump_dir() and 0 <= _gi < 3:
            _dsv41_gdump_tensor(f"F{_gi:02d}in", x)
        x = self.ffn(x, input_ids)
        if _dsv41_gdump_dir() and 0 <= _gi < 3:
            _dsv41_gdump_tensor(f"F{_gi:02d}out", x)
        return x, residual, post_mix, res_mix, ffn_pre'''
assert src.count(old_ffn) == 1, src.count(old_ffn)
src = src.replace(old_ffn, new_ffn, 1)
open(path, "w").write(src); print("patched", path)
