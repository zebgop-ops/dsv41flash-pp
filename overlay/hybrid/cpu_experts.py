# SPDX-License-Identifier: Apache-2.0
"""Routed experts of selected layers computed on the CPU (kt-kernel, AVX2 MXFP4).

For every layer id in DSV41_CPU_EXPERT_LAYERS the DeepseekV4MoE keeps its GPU
router, shared expert and hyper-connection plumbing; only the 384 routed
experts are served by kt-kernel from the native MXFP4 shards (the same bytes
vLLM would have repacked for Marlin), so no re-quantization happens.

Flow per layer step (all on the current CUDA stream):
  router.select_experts (GPU)  ->  D2H hidden/topk into pinned buffers
  ->  kt-kernel forward (host callback / stream-ordered)  ->  H2D output.

kt-kernel requires the caller to keep the batch under `chunked_prefill_size`;
we size it from max_num_batched_tokens so chunked prefill can never overrun.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


import contextlib
import re as _re
import types


@contextlib.contextmanager
def cpu_expert_layer_ctx(prefix: str):
    """Create the routed experts of a CPU layer on the meta device (no HBM)."""
    m = _re.search(r"layers\.(\d+)\.", prefix)
    if m and int(m.group(1)) in cpu_expert_layers():
        with torch.device("meta"):
            yield
    else:
        yield


def is_cpu_expert_weight(name: str) -> bool:
    if ".experts." not in name or ".shared_experts." in name:
        return False
    m = _re.search(r"layers\.(\d+)\.", name)
    return bool(m) and int(m.group(1)) in cpu_expert_layers()


class _QuantMethodProxy:
    """Exposes the original MoE quant method's attributes (the runner reads
    skip_forward_padding, topk_indices_dtype, is_monolithic, ...) without being
    a QuantizeMethodBase instance, so the loader's post-load pass skips it."""

    def __init__(self, orig):
        object.__setattr__(self, "_orig", orig)

    def process_weights_after_loading(self, layer):  # no GPU weights to prepare
        return None

    def apply(self, *a, **k):  # pragma: no cover
        raise RuntimeError("routed experts of this layer run on the CPU")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_orig"), name)


class _StubRoutedExperts(torch.nn.Module):
    """Replaces the meta-device RoutedExperts so the loader's post-load hooks
    and the unloaded-parameter check never see them."""

    def __init__(self, orig):
        super().__init__()
        self.quant_method = _QuantMethodProxy(orig.quant_method)
        for attr in ("activation", "moe_config", "expert_map_manager"):
            if hasattr(orig, attr):
                try:
                    object.__setattr__(self, attr, getattr(orig, attr))
                except Exception:
                    pass

    def _ensure_moe_quant_config_init(self):
        return None

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError("routed experts of this layer run on the CPU")


def cpu_expert_layers() -> set[int]:
    spec = os.environ.get("DSV41_CPU_EXPERT_LAYERS", "").strip()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


class CpuRoutedExperts:
    """Thin owner of one kt-kernel KTMoEWrapper for one layer."""

    _shared_pool_threads: int | None = None

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        weight_path: str,
        max_tokens: int,
        swiglu_limit: float,
        gpu_experts_mask: torch.Tensor | None = None,
    ) -> None:
        threads = int(os.environ.get("DSV41_CPU_EXPERT_THREADS", "16"))
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.num_experts = num_experts
        self.backend = os.environ.get("DSV41_CPU_MOE", "native")
        if self.backend == "native":
            # overlay/hybrid/cpu_moe.cpp: AVX2/OpenMP MXFP4 experts straight from the
            # HF shards; ~2x kt-kernel at batch 1 (which runs one thread per expert).
            from hybrid.cpu_moe import NativeCpuMoe

            self.native = NativeCpuMoe(
                weight_path, layer_idx, num_experts, hidden_size, intermediate_size,
                swiglu_limit, threads=threads, max_tokens=max_tokens,
            )
            self.wrapper = None
            logger.info(
                "layer %d: %d routed experts on CPU via native MXFP4 kernel (%d threads, "
                "max_tokens %d, swiglu_limit %s)",
                layer_idx, num_experts, threads, max_tokens, swiglu_limit,
            )
            return
        from kt_kernel import KTMoEWrapper

        self.wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_gpu_experts=0,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=threads,
            threadpool_count=1,
            weight_path=weight_path,
            chunked_prefill_size=max_tokens,
            method="MXFP4",
            cpu_save=False,
            max_deferred_experts_per_token=0,
            swiglu_limit=swiglu_limit,
        )
        logger.info(
            "layer %d: %d routed experts on CPU via kt-kernel MXFP4 (%d threads, "
            "max_tokens %d, swiglu_limit %s)",
            layer_idx,
            num_experts,
            threads,
            max_tokens,
            swiglu_limit,
        )
        self.num_experts = num_experts

    def load_weights(self) -> None:
        import gc

        if self.wrapper is None:
            self.native.load()
            dev = torch.device("cuda", torch.cuda.current_device())
            self.native.warmup(dev, self.top_k)
            logger.info("layer %d: CPU experts loaded (native)", self.layer_idx)
            return
        p2l = torch.arange(self.num_experts, dtype=torch.int64)
        self.wrapper.load_weights(p2l)
        # kt-kernel's C++ MoE memcpys the packed FP4 weights and converts the
        # scales into its own aligned buffers, but the Python wrapper keeps its
        # contiguous copies alive on the MXFP4 path -> 2x host RAM per layer.
        for attr in ("gate_weights", "up_weights", "down_weights",
                     "gate_scales", "up_scales", "down_scales",
                     "gate_proj", "up_proj", "down_proj"):
            if hasattr(self.wrapper, attr):
                try:
                    setattr(self.wrapper, attr, None)
                except Exception:
                    pass
        gc.collect()
        logger.info("layer %d: CPU experts loaded; python-side copies released",
                    self.layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.wrapper is None:
            return self.native.forward(
                hidden_states.view(-1, hidden_states.shape[-1]), topk_ids, topk_weights
            )
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        return self.wrapper.forward(
            hidden_states.view(-1, hidden_states.shape[-1]),
            topk_ids,
            topk_weights,
            stream,
        )


def install_cpu_experts(model: torch.nn.Module, vllm_config) -> None:
    layer_ids = cpu_expert_layers()
    if not layer_ids:
        return
    # kt-kernel keeps only ONE temporary pinned I/O buffer set per batch size
    # unless the size is registered as a capture size; a CUDA graph captured
    # with a temporary buffer replays against freed pinned memory (segfault in
    # the worker thread). Register vLLM's capture sizes so those buffers persist.
    try:
        if os.environ.get("DSV41_CPU_MOE", "native") != "kt":
            raise ImportError("native CPU MoE backend: no kt-kernel capture sizes needed")
        from kt_kernel import KTMoEWrapper as _KT

        sizes = sorted(set(int(s) for s in (vllm_config.compilation_config.cudagraph_capture_sizes or [])))
        if sizes:
            _KT.set_capture_batch_sizes(sizes)
            logger.info("kt-kernel capture batch sizes registered: %s", sizes)
    except Exception as e:  # pragma: no cover
        logger.warning("could not register kt-kernel capture batch sizes: %s", e)
    """Rewire the MoE runner of each listed layer to compute routed experts on
    the CPU. Must run after model construction (the experts' GPU weights were
    created on the meta device by `meta_experts_context`) and before the
    weight loader (which skips those tensors)."""
    from vllm.model_executor.models.utils import extract_layer_index

    config = vllm_config.model_config.hf_config
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    model_dir = vllm_config.model_config.model
    for name, module in model.named_modules():
        if not name.endswith(".ffn") or not hasattr(module, "experts"):
            continue
        layer_idx = extract_layer_index(name)
        if layer_idx not in layer_ids:
            continue
        runner = module.experts
        cpu = CpuRoutedExperts(
            layer_idx=layer_idx,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            weight_path=model_dir,
            max_tokens=max_tokens,
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0) or 0.0),
        )
        runner._dsv41_cpu_experts = cpu
        runner.routed_experts = _StubRoutedExperts(runner.routed_experts)
        runner._forward_impl = types.MethodType(_cpu_forward_impl, runner)
        logger.info("layer %d: routed experts rewired to CPU", layer_idx)


def load_cpu_expert_weights(model: torch.nn.Module) -> None:
    """Read the CPU layers' experts from the shards (call after the GPU load)."""
    for module in model.modules():
        cpu = getattr(module, "_dsv41_cpu_experts", None)
        if cpu is not None:
            cpu.load_weights()


def _cpu_forward_impl(self, hidden_states, router_logits, shared_experts_input,
                      input_ids=None):
    """MoERunner._forward_impl for a CPU-expert layer: gate on the GPU, route on
    the GPU, experts on the CPU, shared expert on the GPU."""
    cpu = self._dsv41_cpu_experts
    if self.gate is not None:
        router_logits, _ = self.gate(hidden_states)
    # The SharedExperts wrapper only computes for the order its kernel picked;
    # run the shared expert MLP directly (GPU) while the CPU experts work.
    shared = (
        self._shared_experts._layer(shared_experts_input)
        if self._shared_experts is not None
        else None
    )
    topk_weights, topk_ids = self.router.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_indices_dtype=torch.int32,
        input_ids=input_ids,
    )
    # Padded rows (CUDA-graph batch padding) must not reach the CPU kernel: they
    # cost a full expert pass each and change which rows share an expert (and
    # therefore the accumulation path) for the real rows. The runner publishes a
    # static per-step mask in the forward context (patch_pad_mask.py).
    try:
        from vllm.forward_context import get_forward_context as _gfc

        _pad = getattr(_gfc(), "is_padding", None)
    except Exception:
        _pad = None
    if _pad is not None and _pad.shape[0] >= topk_ids.shape[0]:
        topk_ids = topk_ids.masked_fill(_pad[: topk_ids.shape[0]].unsqueeze(1), -1)
    routed = cpu.forward(hidden_states, topk_weights.float(), topk_ids)
    routed = routed.view_as(hidden_states)
    return self._maybe_combine(shared, routed)


