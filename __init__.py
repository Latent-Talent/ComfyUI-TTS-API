"""ComfyUI TTS API integration: sidebar and managed local speech service.

This extension registers only one internal queue-reservation node.
"""
WEB_DIRECTORY = "./web"
if __package__:
    from .queue_bridge import TTSQueueSlot
    NODE_CLASS_MAPPINGS = {"LatentTalentTTSQueueSlot": TTSQueueSlot}
    NODE_DISPLAY_NAME_MAPPINGS = {"LatentTalentTTSQueueSlot": "TTS Queue Reservation (Internal)"}
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

if __package__:
    from .managed_service import initialize as _initialize_tts_api
    _initialize_tts_api()

__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
