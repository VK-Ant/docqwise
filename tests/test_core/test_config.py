"""Tests for configuration system."""

from docqwise.config import DocqwiseConfig, StrategyConfig, SpeedConfig


def test_default_config():
    config = DocqwiseConfig()
    assert config.strategy.mode == "auto"
    assert config.strategy.confidence_threshold == 0.85
    assert config.speed.workers == 1
    assert config.storage.vector_store == "sqlite"
    assert config.search.mode == "vector"
    assert config.incremental.enabled is True
    assert config.chunker.strategy == "structure"


def test_config_custom():
    config = DocqwiseConfig(
        strategy=StrategyConfig(mode="hybrid", confidence_threshold=0.9),
        speed=SpeedConfig(workers=8, gpu_workers=2),
    )
    assert config.strategy.mode == "hybrid"
    assert config.speed.workers == 8
    assert config.speed.gpu_workers == 2


def test_engine_creation():
    from docqwise import Docqwise

    dq = Docqwise()
    assert repr(dq) == "Docqwise(strategy='auto', store='sqlite', workers=1)"

    dq2 = Docqwise(strategy="hybrid", workers=8)
    assert "hybrid" in repr(dq2)


def test_engine_metrics():
    from docqwise import Docqwise

    dq = Docqwise()
    m = dq.metrics()
    assert m["version"] == "0.4.0"
    assert m["strategy"] == "auto"
