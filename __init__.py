"""ComfyUI-TTS-API node entrypoint."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .managed_service import initialize as _initialize_tts_service

_initialize_tts_service()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
