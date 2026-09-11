#!/usr/bin/env python3
"""Patch vLLM's deepseek_v4_1/common/engram.py for DSV41_ENGRAM_STORAGE=ssd.

Idempotent anchor-based edits (run on the image's pristine copy):
  1. ParallelEngramEmbedding.__init__: when storage == "ssd", allocate no table;
     the weight/scale params become empty and their loader only validates shapes.
  2. ParallelEngramEmbedding.lookup: route to SsdEngramTable (host callback + H2D +
     torch dequant) instead of the UVA Triton gather.
  3. `Engram.__init__` passes layer id + model dir so the table can find its shard.
usage: patch_engram.py <path/to/engram.py>
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()
if "DSV41_ENGRAM_SSD" in src:
    print("already patched"); sys.exit(0)

# --- 1. imports + storage helper -------------------------------------------
anchor = "logger = init_logger(__name__)\n"
assert anchor in src
src = src.replace(anchor, anchor + '''
# DSV41_ENGRAM_SSD: tables served from the checkpoint shards on the NVMe (overlay).
import os as _os


def _engram_storage() -> str:
    return _os.environ.get("DSV41_ENGRAM_STORAGE", "host").strip().lower()
''', 1)

# --- 2. constructor: skip the table when storage == ssd ----------------------
old = '''        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
'''
new = '''        self.ssd_table = None
        self._ssd = _engram_storage() == "ssd"
        if self._ssd:
            cpu_offload = False
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
'''
assert old in src; src = src.replace(old, new, 1)

old = '''        # Explicit device: model init runs under a `torch.device("cuda")`
        # context, which would otherwise put the shard in HBM.
        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.weight = nn.Parameter(
            torch.empty(
                self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn, **kwargs
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                dim // block_size,
                dtype=torch.uint8,
                **kwargs,
            ),
            requires_grad=False,
        )
'''
new = '''        # Explicit device: model init runs under a `torch.device("cuda")`
        # context, which would otherwise put the shard in HBM.
        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        table_rows = 0 if self._ssd else self.part_num_embeddings
        self.weight = nn.Parameter(
            torch.empty(
                table_rows, dim, dtype=torch.float8_e4m3fn, **kwargs
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                table_rows,
                dim // block_size,
                dtype=torch.uint8,
                **kwargs,
            ),
            requires_grad=False,
        )
        if self._ssd:
            # The loader must never copy the 95 GiB table: validate and drop.
            for param in (self.weight, self.weight_scale_inv):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": _engram_ssd_weight_loader,
                        "engram_expected_rows": num_embeddings,
                    },
                )
'''
assert old in src; src = src.replace(old, new, 1)

old = '''        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _engram_head_shard_weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )
        if cpu_offload:
'''
new = '''        if not self._ssd:
            for param in (self.weight, self.weight_scale_inv):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": _engram_head_shard_weight_loader,
                        "engram_vocab_start": self.vocab_start_idx,
                    },
                )
        if cpu_offload:
'''
assert old in src; src = src.replace(old, new, 1)

# --- 3. ssd loader + attach() + lookup routing ------------------------------
old = '''def _engram_head_shard_weight_loader(
'''
new = '''def _engram_ssd_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """SSD storage: the table stays in the shard; only check the checkpoint shape."""
    rows = param.engram_expected_rows
    if loaded_weight.shape[0] != rows:
        raise ValueError(
            f"engram table has {loaded_weight.shape[0]} rows, config says {rows}"
        )


def _engram_head_shard_weight_loader(
'''
assert old in src; src = src.replace(old, new, 1)

old = '''    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
'''
new = '''    def attach_ssd(self, model_dir: str, layer_id: int, max_tokens: int) -> None:
        """SSD storage: open the shard that holds this layer's table."""
        if not self._ssd:
            return
        from engram_ssd.engram_ssd import SsdEngramTable

        self.ssd_table = SsdEngramTable(
            model_dir,
            layer_id,
            self.vocab_start_idx,
            self.vocab_end_idx,
            self.head_start,
            self.part_n_hash_cols,
            self.n_hash_cols,
            dim=self.dim,
            block_size=self.block_size,
            max_tokens=max_tokens,
        )

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if self._ssd:
            assert self.ssd_table is not None, "attach_ssd() not called"
            self.ssd_table.lookup(indices, out)
            return
'''
assert old in src; src = src.replace(old, new, 1)

# --- 4. Engram.__init__: attach the table to its shard ------------------------
import re as _re
_m = _re.search(r"^(        max_tokens = (?:vllm_config|get_current_vllm_config\(\))\.scheduler_config\.max_num_batched_tokens\n)", src, _re.M)
assert _m, "Engram.__init__ max_tokens anchor missing"
src = src.replace(_m.group(1), _m.group(1) + '''        if self.embed_tokens._ssd:
            _model_dir = _os.environ.get("DSV41_MODEL_DIR") or get_current_vllm_config().model_config.model
            self.embed_tokens.attach_ssd(
                _model_dir, layout.layer_ids[layer_hash_index], max_tokens
            )
''', 1)

open(path, "w").write(src)
print("patched", path)
