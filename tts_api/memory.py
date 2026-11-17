"""Best-effort memory cleanup and diagnostics for the isolated TTS worker.

Never imports torch itself: reading /health must not initialize a CUDA context.
CUDA allocated/reserved figures describe this Python process, not the separate
IndexTTS process or ComfyUI's main process.
"""
from __future__ import annotations

import gc
import logging
import sys

LOG = logging.getLogger("comfyui-tts-api.memory")


def release_unused_cuda() -> None:
    """Collect orphaned tensors and return allocator cache, preserving failures."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        # Cleanup must not replace the original model-loading exception.
        LOG.debug("[TTS API] Optional CUDA cache cleanup failed", exc_info=True)


def process_cuda_memory() -> dict:
    """Return per-device PyTorch allocator usage without importing torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"available": False, "reason": "torch_not_imported"}
    try:
        if not torch.cuda.is_initialized():
            return {"available": False, "reason": "cuda_not_initialized"}
        devices = []
        for index in range(torch.cuda.device_count()):
            devices.append({
                "device": index,
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            })
        return {"available": True, "devices": devices}
    except Exception:
        return {"available": False, "reason": "cuda_metrics_unavailable"}
