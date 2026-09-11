#!/usr/bin/env python3
"""Publish a static per-step padding mask (rows >= num scheduled tokens in a CUDA-graph
padded batch) as ForwardContext.is_padding, for both real steps and capture dummy runs, so
in-graph consumers (overlay/hybrid/cpu_experts.py) can skip padded rows at replay.
usage: <vllm/v1/worker/gpu_model_runner.py>"""
import re, sys
path = sys.argv[1]; src = open(path).read()
if "_dsv41_pad_set" in src:
    print("already patched"); sys.exit(0)
old_exec = "                slot_mapping=slot_mappings,\n                skip_compiled=has_encoder_input,\n            ),\n"
new_exec = ("                slot_mapping=slot_mappings,\n                skip_compiled=has_encoder_input,\n"
            "                is_padding=self._dsv41_pad_set(num_tokens_unpadded, num_tokens_padded),\n            ),\n")
assert src.count(old_exec) == 1, src.count(old_exec)
src = src.replace(old_exec, new_exec, 1)
old_dummy = "                    slot_mapping=slot_mappings,\n                ),\n            ):\n                outputs = self.model(\n"
new_dummy = ("                    slot_mapping=slot_mappings,\n"
             "                    is_padding=self._dsv41_pad_view(num_tokens_padded),\n                ),\n            ):\n                outputs = self.model(\n")
assert src.count(old_dummy) == 1, src.count(old_dummy)
src = src.replace(old_dummy, new_dummy, 1)
helper = '''    def _dsv41_pad_view(self, num_padded: int) -> torch.Tensor:
        """Static bool buffer [max tokens]: True for CUDA-graph padding rows."""
        buf = self.__dict__.get("_dsv41_pad_buf")
        if buf is None or buf.numel() < num_padded:
            n = max(int(num_padded), int(getattr(self, "max_num_tokens", num_padded)) * 2)
            buf = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.__dict__["_dsv41_pad_buf"] = buf
            self.__dict__["_dsv41_pad_cpu"] = torch.zeros(n, dtype=torch.bool, device="cpu", pin_memory=True)
        return buf[:num_padded]

    def _dsv41_pad_set(self, num_actual: int, num_padded: int) -> torch.Tensor:
        view = self._dsv41_pad_view(num_padded)
        cpu = self.__dict__["_dsv41_pad_cpu"]
        cpu[:num_padded] = False
        if num_padded > num_actual:
            cpu[num_actual:num_padded] = True
        view.copy_(cpu[:num_padded], non_blocking=True)
        return view

    def _init_model_kwargs(self, num_reqs: int | None = None):'''
old = "    def _init_model_kwargs(self, num_reqs: int | None = None):"
assert src.count(old) == 1
src = src.replace(old, helper, 1)
open(path, "w").write(src); print("patched", path)
