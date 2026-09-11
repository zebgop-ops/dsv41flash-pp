#!/usr/bin/env python3
"""PP (non-async) + speculative decoding token bookkeeping.

The scheduler sends non-last PP ranks only the non-draft token scheduled this step, but after a
step with accepted drafts the worker is also missing the accepted draft tokens: its
`num_new_tokens` formula grows past the one token it received, numpy broadcasts that token over
the accepted positions in token_ids_cpu (corrupting the history the Engram lookback hashes),
and the slice can come back empty (`new_token_ids[-1]` IndexError on rank 0). Fix: the
scheduler tracks how far each request's tokens were sent to the workers and sends everything
from there up to the current non-draft token; the worker appends exactly what it receives.
usage: patch_pp_spec_tokens.py <vllm/v1/core/sched/scheduler.py> <vllm/v1/worker/gpu_model_runner.py>"""
import sys
sched, runner = sys.argv[1], sys.argv[2]
s = open(sched).read()
if "dsv41: send every token" not in s:
    old = '''                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)'''
    new = '''                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                # dsv41: send every token the workers have not seen yet (accepted
                # draft tokens included), not just the one scheduled this step.
                end = req.num_computed_tokens + num_tokens
                sent = max(self._dsv41_pp_sent.get(req_id, 0), req.num_prompt_tokens)
                token_ids = req.all_token_ids[sent:end]
                self._dsv41_pp_sent[req_id] = max(sent, end)
                new_token_ids.append(token_ids)'''
    assert s.count(old) == 1, s.count(old); s = s.replace(old, new, 1)
    old = "    def _free_request("
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, '''    @property
    def _dsv41_pp_sent(self) -> dict[str, int]:
        d = self.__dict__.get("_dsv41_pp_sent_dict")
        if d is None:
            d = self.__dict__["_dsv41_pp_sent_dict"] = {}
        return d

''' + old, 1)
    # drop the tracker when a request is freed
    i = s.index("    def _free_request(")
    j = s.index(":\n", s.index(")", i)) + 2
    s = s[:j] + "        self._dsv41_pp_sent.pop(request.request_id, None)\n" + s[j:]
    # PP batch queue schedules ahead: a request whose latest real token is still in
    # flight must not be scheduled again (its freshly proposed drafts would be the only
    # tokens scheduled and the logits indices go negative / wrap).
    old3 = '''            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:'''
    new3 = '''            # dsv41: under PP (non-async) the engine schedules ahead while the
            # previous step is in flight; only drafts would be schedulable for a
            # request whose real tokens are all computed -> wait for its output.
            if (
                self.use_pp
                and not self.scheduler_config.async_scheduling
                and request.num_tokens + request.num_output_placeholders
                <= request.num_computed_tokens
            ):
                req_index += 1
                continue
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:'''
    assert s.count(old3) == 1, s.count(old3)
    s = s.replace(old3, new3, 1)
    open(sched, "w").write(s); print("patched", sched)
else:
    print("already patched", sched)
r = open(runner).read()
if "dsv41: append exactly" not in r:
    old = '''                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )'''
    new = '''                    new_token_ids = req_data.new_token_ids[i]
                    # dsv41: append exactly what the scheduler sent (it sends every
                    # output token the worker has not seen, drafts accepted last
                    # step included; see patch_pp_spec_tokens.py).
                    if new_token_ids:
                        # Re-sent tokens (re-prefill after preemption) are dropped:
                        # the scheduler's slice ends at the current non-draft token.
                        _end = num_computed_tokens + scheduler_output.num_scheduled_tokens[
                            req_id
                        ] - len(scheduled_spec_tokens.get(req_id, ()))
                        _drop = max(0, req_state.num_tokens - (_end - len(new_token_ids)))
                        if _drop:
                            new_token_ids = new_token_ids[_drop:]
                        if new_token_ids:
                            req_state.output_token_ids.extend(new_token_ids)'''
    assert r.count(old) == 1, r.count(old); r = r.replace(old, new, 1)
    old = '''                end_token_index = max(
                    start_token_index,
                    num_computed_tokens + len(new_token_ids),
                )
                if end_token_index > start_token_index:
                    if new_token_ids:
                        # Add new_token_ids to token_ids_cpu.
                        num_new_tokens = end_token_index - start_token_index
                        tokens_to_append = new_token_ids[-num_new_tokens:]
                        self.input_batch.token_ids_cpu[
                            req_index, start_token_index:end_token_index
                        ] = tokens_to_append'''
    new = '''                if new_token_ids:
                    # dsv41: the received tokens are exactly the ones missing
                    # from [start_token_index, ...).
                    end_token_index = start_token_index + len(new_token_ids)
                else:
                    end_token_index = max(
                        start_token_index,
                        num_computed_tokens + len(new_token_ids),
                    )
                if end_token_index > start_token_index:
                    if new_token_ids:
                        # Add new_token_ids to token_ids_cpu.
                        self.input_batch.token_ids_cpu[
                            req_index, start_token_index:end_token_index
                        ] = new_token_ids'''
    assert r.count(old) == 1, r.count(old); r = r.replace(old, new, 1)
    open(runner, "w").write(r); print("patched", runner)
else:
    print("already patched", runner)
