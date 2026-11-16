# ComfyUI TTS API

Local, modular speech synthesis for **ComfyUI** and **SillyTavern**.

The first version runs a lightweight FastAPI gateway in an isolated virtual environment and routes audio to vLLM-Omni model servers. VoxCPM2 is the default speech/voice-cloning backend. Qwen3-TTS supports description-based voice creation and reusable clones.

Full installation, security, API and usage instructions are included below.

> Model weights and vLLM-Omni are not bundled; install and start compatible upstream model servers separately. Hardware inference has not been benchmarked as part of this repository scaffold.

