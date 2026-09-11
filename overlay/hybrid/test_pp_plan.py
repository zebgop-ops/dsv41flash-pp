"""CPU-only check of the PP shadow plan against the V4.1 config (vllm stubs)."""
import sys, types, json
# stub vllm.distributed / logger
class _PP: 
    def __init__(s, r, w): s.rank_in_group, s.world_size = r, w
stubs = {}
def mk(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
mk("vllm"); mk("vllm.logger", init_logger=lambda n: types.SimpleNamespace(info=lambda *a, **k: None))
def get_pp_indices(n, r, w):
    import os
    parts = [int(x) for x in os.environ["VLLM_PP_LAYER_PARTITION"].split(",")]
    assert sum(parts) == n and len(parts) == w
    start = sum(parts[:r]); return start, start + parts[r]
mk("vllm.distributed.utils", get_pp_indices=get_pp_indices)
dist = mk("vllm.distributed", get_pp_group=lambda: _PP(CUR[0], 4))
import torch  # noqa
CUR = [0]
import os; os.environ["VLLM_PP_LAYER_PARTITION"] = sys.argv[1] if len(sys.argv) > 1 else "8,6,10,16"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from hybrid import pp_shadow
cfg = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-1000/-home-r-claude/797f92eb-7c73-445f-bced-0f32d24e05f0/scratchpad/config.json"))["text_config"]
cfg = types.SimpleNamespace(**cfg)
for r in range(4):
    CUR[0] = r
    start, end = get_pp_indices(cfg.num_hidden_layers, r, 4)
    plan = pp_shadow.PPShadowPlan(cfg, cfg.num_hidden_layers, start, end)
    print(f"rank {r} [{start},{end}): shadow={plan.shadow_ids} ship={plan.ship_ids} owned={plan.owned_ship_ids} recv={plan.recv_ids} cand ship/recv={plan.ship_cand}/{plan.recv_cand}")
