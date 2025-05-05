from pathlib import Path

import pytest

from darts_segmentation.training.hparams import parse_hyperparameters

TEST_HP_CONTENT_YAML = """
hyperparameters:
    learning_rate:
        distribution: loguniform
        low: 1.0e-5
        high: 1.0e-2
    batch_size:
        distribution: choice
        choices: [32, 64, 128]
    gamma:
        distribution: uniform
        low: 0.9
        high: 2.5
    dropout:
        distribution: uniform
        low: 0.0
        high: 0.5
    layers:
        distribution: intuniform
        low: 1
        high: 10
    architecture:
        distribution: constant
        value: "resnet"
"""

TEST_HP_CONTENT_TOML = """
[hyperparameters]
learning_rate = {distribution = "loguniform", low = 1.0e-5, high = 1.0e-2}
batch_size = {distribution = "choice", choices = [32, 64, 128]}
gamma = {distribution = "uniform", low = 0.9, high = 2.5}
dropout = {distribution = "uniform", low = 0.0, high = 0.5}
layers = {distribution = "intuniform", low = 1, high = 10}
architecture = {distribution = "constant", value = "resnet"}
"""

TEST_HP_CONTENT_YAML_IMPLICIT = """
hyperparameters:
    learning_rate:
        distribution: loguniform
        low: 1.0e-5
        high: 1.0e-2
    batch_size: [32, 64, 128]
    gamma:
        low: 0.9
        high: 2.5
    dropout:
        low: 0.0
        high: 0.5
    layers:
        low: 1
        high: 10
    architecture: "resnet"
"""


@pytest.mark.parametrize(
    "content",
    [
        (TEST_HP_CONTENT_YAML_IMPLICIT, "yaml"),
        (TEST_HP_CONTENT_YAML, "yaml"),
        (TEST_HP_CONTENT_TOML, "toml"),
    ],
)
def test_parse_hparams(content: tuple[str, str], tmp_path: Path):
    text, suffix = content
    hpconfig_file = tmp_path / f"hparams.{suffix}"
    hpconfig_file.write_text(text)
    hparams = parse_hyperparameters(hpconfig_file)

    assert isinstance(hparams, dict)

    # Assert that all values are either an object with a `rvs` method or a list
    for distribution in hparams.values():
        assert isinstance(distribution, list) or hasattr(distribution, "rvs")

    # Test learning_rate
    samples = hparams["learning_rate"].rvs(size=1000)
    assert all(1e-5 <= sample <= 1e-2 for sample in samples)

    # Test batch_size
    assert len(hparams["batch_size"]) == 3
    assert all(sample in [32, 64, 128] for sample in hparams["batch_size"])

    # Test gamma
    samples = hparams["gamma"].rvs(size=1000)
    assert all(0.9 <= sample <= 2.5 for sample in samples)

    # Test dropout
    samples = hparams["dropout"].rvs(size=1000)
    assert all(0.0 <= sample <= 0.5 for sample in samples)

    # Test layers
    samples = hparams["layers"].rvs(size=1000)
    assert all(1 <= sample <= 10 for sample in samples)
    assert samples.min() == 1
    assert samples.max() == 10

    # Test architecture
    assert hparams["architecture"][0] == "resnet"
    assert len(hparams["architecture"]) == 1
