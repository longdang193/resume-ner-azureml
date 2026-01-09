"""
Unit tests to assert defaults defined in config/train.yaml are parsed as expected.
"""

from pathlib import Path
import yaml


def _load_train_yaml():
    root = Path(__file__).resolve().parents[3]
    train_yaml = root / "config" / "train.yaml"
    return yaml.safe_load(train_yaml.read_text())


def test_core_training_defaults():
    cfg = _load_train_yaml()
    training = cfg["training"]
    assert training["epochs"] == 1
    assert training["batch_size"] == 1
    assert float(training["learning_rate"]) == 2e-5
    assert float(training["weight_decay"]) == 0.01
    assert training["warmup_steps"] == 500
    assert float(training["max_grad_norm"]) == 1.0
    assert training["gradient_accumulation_steps"] == 2


def test_metric_defaults():
    cfg = _load_train_yaml()
    training = cfg["training"]
    assert training["metric"] == "macro-f1"
    assert training["metric_mode"] == "max"


def test_early_stopping_defaults():
    cfg = _load_train_yaml()
    early = cfg["training"]["early_stopping"]
    assert early["enabled"] is True
    assert early["patience"] == 3
    assert early["min_delta"] == 0.001

