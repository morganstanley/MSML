from harness.config import DataConfig, ModelConfig, OptimConfig, TrainConfig, ExperimentConfig


def test_defaults_sane():
    cfg = ExperimentConfig()
    assert cfg.train.time_limit_seconds == 1200
    assert cfg.train.param_cap == 100_000_000
    assert cfg.data.batch_size > 0
    assert cfg.data.block_size > 0
    assert cfg.optim.lr > 0
    assert cfg.optim.weight_decay >= 0


def test_override_fields_at_construction():
    dc = DataConfig(batch_size=4, block_size=32, seed=1)
    mc = ModelConfig(n_layer=2, n_head=2, n_embd=64)
    oc = OptimConfig(lr=1e-3)
    tc = TrainConfig(time_limit_seconds=30, param_cap=10_000_000, device="cpu")
    cfg = ExperimentConfig(data=dc, model=mc, optim=oc, train=tc, run_name="x", out_dir="y")

    assert cfg.data.batch_size == 4
    assert cfg.data.block_size == 32
    assert cfg.model.n_layer == 2
    assert cfg.optim.lr == 1e-3
    assert cfg.train.time_limit_seconds == 30
    assert cfg.train.param_cap == 10_000_000
    assert cfg.train.device == "cpu"


def test_param_cap_is_configurable_field():
    cfg = ExperimentConfig()
    cfg.train.param_cap = 123
    assert cfg.train.param_cap == 123
