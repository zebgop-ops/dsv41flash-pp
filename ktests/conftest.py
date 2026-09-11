import pytest

@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        yield cfg
