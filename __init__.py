"""ComfyUI custom-node entrypoint.

ComfyUI imports this as a package; ordinary test runners sometimes import the
repository's __init__.py as a top-level module instead.
"""
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .managed_service import initialize as _initialize_tts_service

    _initialize_tts_service()
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
