"""ComfyUI TTS API integration: sidebar and managed local speech service.

This extension intentionally registers NO ComfyUI workflow nodes.
"""
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if __package__:
    from .managed_service import initialize as _initialize_tts_api
    _initialize_tts_api()

__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
