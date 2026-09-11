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
open(path, "w").write(src); print("patched", path)
