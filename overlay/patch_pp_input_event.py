#!/usr/bin/env python3
"""Guard the runner's reused pinned input buffers under pipeline parallel even without async scheduling.
vLLM creates `prepare_inputs_event` (wait for the previous step's async H2D copies before rewriting the CPU
staging tensors) only for async scheduling. Under PP the engine's batch queue keeps several batches in
flight per rank in synchronous mode too (prefill-chunk pipelining, deferred draft fetch): a rank's stream
can still be blocked on an upstream receive for batch k-1 when the CPU prepares batch k+1, and the not yet
executed copies of batch k then read batch k+1's values (seen as a device-side assert in repeat_interleave:
GPU query_start_loc total != CPU total).
usage: <v1/worker/gpu_model_runner.py>"""
import sys
p = sys.argv[1]; s = open(p).read()
if "dsv41 pp input event" in s:
    print("already patched"); sys.exit(0)
old = '''        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
            # under TP contention that spin can balloon and make the rank a straggler.
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)
'''
new = old + '''        elif self.parallel_config.pipeline_parallel_size > 1:  # dsv41 pp input event
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)
'''
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new)); print("patched", p)
