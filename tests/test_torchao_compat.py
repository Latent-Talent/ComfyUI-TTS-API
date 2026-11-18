"""Select TorchAO by the worker's installed PyTorch, without touching CUDA."""
from __future__ import annotations

import pytest

from tts_api.torchao_compat import torchao_for_torch


@pytest.mark.parametrize(("torch", "spec"), [
    ("2.8.0+cu128", "torchao==0.13.0"),
    ("2.8.0", "torchao==0.13.0"),
    ("2.9.0+cu128", "torchao==0.14.1"),
    ("2.9.1+cu128", "torchao==0.15.0"),
    ("2.10.0", "torchao==0.16.0"),
    ("2.11.0+cu130", "torchao==0.18.0"),
    ("2.12.1+cu130", "torchao==0.18.0"),
])
def test_torchao_version_compatibility(torch, spec):
    assert torchao_for_torch(torch) == spec


@pytest.mark.parametrize("version", [
    "bad", "2.7.1", "1.13.1", "3.0.0",
])
def test_unsupported_torch_is_not_silently_upgraded(version):
    with pytest.raises(ValueError):
        torchao_for_torch(version)
