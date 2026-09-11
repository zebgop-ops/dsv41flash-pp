#!/usr/bin/env python3
"""Mxfp4MoEMethod: stage the Marlin repack through host RAM (hybrid/marlin_staged.py) so the
load-time HBM peak is the packed weight only, not raw + packed (~6.7 GiB/layer). Also stops
process_weights_after_loading/_setup_kernel from pinning the raw tensors in their frames.
usage: <vllm/model_executor/layers/quantization/mxfp4.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: staged marlin" in src:
    print("already patched"); sys.exit(0)
# only the second (Mxfp4MoEMethod) occurrence: split on the class header
head, tail = src.split("class Mxfp4MoEMethod(FusedMoEMethodBase):", 1)
old = '''        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format('''
new = '''        # dsv41: staged marlin repack (no raw+packed HBM transient)
        if self.mxfp4_backend == Mxfp4MoeBackend.MARLIN:
            del w13, w2, w13_scale, w2_scale  # the layer keeps the only references
            from hybrid.marlin_staged import prepare_marlin_mxfp4_moe_staged

            prepare_marlin_mxfp4_moe_staged(layer)
            w13, w2 = layer.w13_weight, layer.w2_weight
            w13_scale, w2_scale = layer.w13_weight_scale, layer.w2_weight_scale
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
        else:
          w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format('''
assert tail.count(old) == 1, tail.count(old)
tail = tail.replace(old, new, 1)
# the convert call's closing paren line must match the new indentation: leave as is (python
# allows any consistent indentation inside the else block since the statement is one expression)
old2 = '''    def process_weights_after_loading(self, layer):
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)'''
new2 = '''    def process_weights_after_loading(self, layer):
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        # dsv41: pass the raw tensors without local references (see _setup_kernel)
        self._setup_kernel(layer, layer.w13_weight, layer.w2_weight, layer.w13_weight_scale,
                           layer.w2_weight_scale, w13_bias, w2_bias)'''
assert tail.count(old2) == 1, tail.count(old2)
tail = tail.replace(old2, new2, 1)
open(path, "w").write(head + "class Mxfp4MoEMethod(FusedMoEMethodBase):" + tail); print("patched", path)
