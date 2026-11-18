"""TorchAO release selection without importing Torch or loading CUDA.

Version mapping follows pytorch/ao issue #2919. TorchAO 0.17+ has a
stable ABI floor at Torch 2.11; newer Torch versions still need runtime
smoke tests for the selected quantization kernels.
"""
from __future__ import annotations

import re


def torchao_for_torch(torch_version: str) -> str:
    """Select a tested/reported TorchAO release family for installed Torch."""
    match = re.match(r"^([0-9]+)\.([0-9]+)\.([0-9]+)(?:\b|[+.-])", torch_version)
    if not match:
        raise ValueError(f"Unrecognized PyTorch version: {torch_version!r}")
    major, minor, patch = (int(x) for x in match.groups())
    if major != 2 or minor < 8:
        raise ValueError(
            f"Unsupported PyTorch {torch_version}. This INT8 installer supports Torch 2.8+."
        )
    if minor == 8:
        return "torchao==0.13.0"
    if minor == 9:
        return "torchao==0.14.1" if patch == 0 else "torchao==0.15.0"
    if minor == 10:
        return "torchao==0.16.0"
    return "torchao==0.18.0"
