#!/usr/bin/env python3
"""CPU routed experts (kt-kernel) for selected layers: create those layers' routed
expert weights on the meta device, skip them in the checkpoint loader, and install
the CPU forward path after construction. Anchor-based, idempotent.
usage: patch_cpu_experts.py <vllm site dir>"""
import os, sys
site = sys.argv[1]

def edit(rel, repls, done):
    p = os.path.join(site, rel); src = open(p).read()
    if done in src:
        print("already patched", rel); return
    for old, new in repls:
        assert old in src, f"anchor missing in {rel}: {old[:70]!r}"
        src = src.replace(old, new, 1)
    open(p, "w").write(src); print("patched", rel)

# 1. deepseek_v4/nvidia/model.py: meta-device routed experts for CPU layers
edit("models/deepseek_v4/nvidia/model.py", [
    ("from vllm.model_executor.models.utils import (\n",
     "from hybrid.cpu_experts import cpu_expert_layer_ctx as _dsv41_cpu_expert_layer_ctx\n"
     "from vllm.model_executor.models.utils import (\n"),
    ('''        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
''', '''        with _dsv41_cpu_expert_layer_ctx(prefix):
          self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
'''),
], "_dsv41_cpu_expert_layer_ctx")

# 2. deepseek_v4_1/nvidia/model.py: loader skip + install hook
edit("models/deepseek_v4_1/nvidia/model.py", [
    ("from vllm.v1.attention.backends.registry import AttentionBackendEnum\n",
     "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n"
     "from hybrid.cpu_experts import (\n"
     "    install_cpu_experts as _dsv41_install_cpu_experts,\n"
     "    is_cpu_expert_weight as _dsv41_is_cpu_expert_weight,\n"
     "    load_cpu_expert_weights as _dsv41_load_cpu_expert_weights,\n"
     ")\n"),
    ('''        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
''', '''        for name, loaded_weight in weights:
            if _dsv41_is_cpu_expert_weight(name):
                continue  # served by kt-kernel straight from the shards
            if name.startswith(("vision.", "aligner.", "image_")):
'''),
    ('''        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
''', '''        self.set_moe_parameters()
        _dsv41_install_cpu_experts(self, vllm_config)

    def set_moe_parameters(self) -> None:
'''),
    ('''    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()
''', '''    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()
        _dsv41_load_cpu_expert_weights(self)
'''),
], "_dsv41_is_cpu_expert_weight")
