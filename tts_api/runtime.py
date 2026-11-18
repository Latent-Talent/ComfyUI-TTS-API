"""Optional CPU/GPU execution and selective TorchAO W8A8 conversion.

Imports of torch and torchao are deliberately lazy: importing the HTTP API
or running its unit tests must not initialize CUDA or require torchao.
The source checkpoints remain untouched; conversion happens at model load.
"""
from __future__ import annotations

import os

RUNTIME_FIELDS = frozenset({"runtime_device", "runtime_precision", "runtime_cpu_threads"})
DEVICE_OPTIONS = ("auto", "cpu", "cuda")
PRECISION_OPTIONS = ("native", "int8")


def runtime_options(settings: dict) -> dict:
    """Read strictly validated runtime defaults without forwarding to speech."""
    device = settings.get("runtime_device", "auto")
    precision = settings.get("runtime_precision", "native")
    threads = settings.get("runtime_cpu_threads", 0)
    if device not in DEVICE_OPTIONS:
        raise ValueError("Unsupported TTS execution device")
    if precision not in PRECISION_OPTIONS:
        raise ValueError("Unsupported TTS model precision")
    if type(threads) is not int or not 0 <= threads <= 256:
        raise ValueError("TTS CPU threads must be 0 (auto) or 1-256")
    return {"device": device, "precision": precision, "cpu_threads": threads}


def resolve_device(selection: str, torch=None) -> str:
    """Fail on an unavailable explicitly selected GPU; never silently switch."""
    if torch is None:
        import torch
    if selection not in DEVICE_OPTIONS:
        raise ValueError("Unsupported TTS execution device")
    available = bool(torch.cuda.is_available())
    if selection == "cuda" and not available:
        raise RuntimeError("TTS CUDA was selected but no CUDA GPU is available")
    return "cuda:0" if (available and selection != "cpu") else "cpu"


def apply_cpu_threads(torch, value: int, device: str) -> None:
    """0 leaves PyTorch defaults intact. Never oversubscribe the CPU affinity."""
    if device != "cpu" or not value:
        return
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if value > available:
        raise ValueError(f"Requested {value} CPU threads but only {available} CPUs are available")
    torch.set_num_threads(value)


def _eligible_linear(module, name: str, torch, scope: str) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    # W8A8 kernels are shape-dependent. Leave tiny and irregular linears native.
    if (module.in_features % 16 or module.out_features % 16 or
            module.weight.numel() < 65536):
        return False
    lowered = name.casefold()
    # Protect acoustic/voice output modules as well as logits/projection heads.
    excluded = ("vocoder", "codec", "decoder", "speaker", "embed",
                "lm_head", "output_head", "classifier", "audio_head")
    if any(component in lowered for component in excluded):
        return False
    if scope == "qwen":
        # Quantize the autoregressive Talker/Code Predictor, never the codec.
        return "talker" in lowered or "predictor" in lowered
    if scope in ("voxcpm", "index"):
        return True
    raise ValueError(f"Unsupported INT8 conversion scope: {scope}")


def require_int8_backend() -> None:
    """Validate the optional INT8 dependency before loading large FP weights."""
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig, quantize_,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "INT8 requires a compatible TorchAO install in the TTS worker. "
            "Run install-int8.sh for this worker or select Native precision."
        ) from exc


def quantize_int8(module, *, scope: str, torch=None) -> int:
    """Convert supported linear weights and dynamic activations to TorchAO W8A8.

    TorchAO owns the kernel selection for CPU/CUDA. No silent fallback to BF16:
    an incompatible model or kernel fails so users can select native precision.
    """
    if torch is None:
        import torch
    require_int8_backend()
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig, quantize_,
    )
    eligible = [
        name for name, layer in module.named_modules()
        if _eligible_linear(layer, name, torch, scope)
    ]
    if not eligible:
        raise RuntimeError(
            f"No compatible linear layers found for {scope} INT8 conversion; "
            "the model has not been quantized."
        )
    allowed = set(eligible)
    from .operations import emit
    backend = {"qwen": "qwen3-tts", "voxcpm": "voxcpm2",
               "index": "indextts"}[scope]
    emit("model.quantization.started", backend=backend, precision="int8",
         quantized_layers=len(eligible))
    try:
        quantize_(
            module, Int8DynamicActivationInt8WeightConfig(),
            filter_fn=lambda layer, name: name in allowed,
        )
    except Exception as exc:
        emit("model.quantization.failed", backend=backend, precision="int8",
             error_type=type(exc).__name__)
        raise RuntimeError(
            f"TorchAO INT8 conversion failed for {scope} ({type(exc).__name__}). "
            "Choose Native precision or use a compatible TorchAO version."
        ) from exc
    emit("model.quantization.completed", backend=backend, precision="int8",
         quantized_layers=len(eligible))
    return len(eligible)
