# ComfyUI TTS API

A **sidebar-managed, local text-to-speech API server** for ComfyUI, with an OpenAI-compatible endpoint for SillyTavern and other clients. Includes VoxCPM2 voice cloning and voice-by-description.

**v0.3 — API + sidebar only. No ComfyUI workflow nodes.** ComfyUI starts the service and manages its GPU memory; voice generation is accessed through the sidebar or HTTP API. This project does not filter, censor, rewrite, or send user text to a moderation service. Model output fidelity is not guaranteed by removing application-side filters.

## Highlights

- Dedicated **TTS API** tab in the native ComfyUI sidebar: status, model memory, restart/unload, character voice library, voice design and clone, and a built-in speech test/audio player.
- A managed local speech API starts automatically when ComfyUI starts, and exits when ComfyUI exits. **No separate `start.sh` needed.**
- VoxCPM2 loads lazily, stays resident between requests, and unloads after 60 seconds idle (configurable).
- When ComfyUI starts an image/video workflow, its prompt handler asks TTS to release VRAM. If the TTS worker cannot yield promptly, ComfyUI terminates its **own managed TTS worker** to free VRAM, interrupting that active TTS request. The service returns after the ComfyUI queue is idle.
- ComfyUI-specific sidebar API requests are proxied through the ComfyUI HTTP server; browser clients do not need access to the internal API key or a separate TTS URL.
- The standalone OpenAI-style speech endpoint provides MP3 by default, WAV, Opus/Ogg, AAC, FLAC, and 24 kHz signed 16-bit PCM.
- Save cloned voices from a reference recording, or design a voice from a natural-language description.
- The TTS runtime uses an **isolated Python virtual environment**, leaving the existing ComfyUI Python/Torch/CUDA packages untouched.
- No cloud account required. Only model weights are downloaded on first use.

**Important:** v0.3 uses the official VoxCPM2 native Python backend. Native 4090 inference, pause/unload integration, and SillyTavern compatibility still require machine-level verification. vLLM-Omni/Nano-vLLM acceleration, Qwen3-TTS, and true streaming audio remain planned, not yet implemented.

## Install or update (WSL / Ubuntu)

Requirements: an existing Python 3.12 executable, `ffmpeg` available on PATH (for MP3), NVIDIA GPU for fast inference, and the main ComfyUI installation.

```bash
cd ~/ComfyUI/custom_nodes
if [ -d ComfyUI-TTS-API ]; then
  git -C ComfyUI-TTS-API pull --ff-only
else
  git clone https://github.com/Latent-Talent/ComfyUI-TTS-API.git
fi
cd ComfyUI-TTS-API
bash install.sh
```

The installer creates `ComfyUI-TTS-API/.venv` and installs the service. This is a **one-time** step; no need to reinstall for ordinary git pulls. It does not change your system Python or the Python environment in `~/ComfyUI/.venv`.

**Restart ComfyUI normally.** Open the **TTS API** icon in ComfyUI's sidebar. You will see three tabs:

- **Server:** online/offline, model VRAM state, idle unload interval, queue state, generation performance, API URL, unload model and restart API.
- **Voices:** list/copy/delete saved voice IDs, create a voice by description, or upload reference audio for cloning.
- **Test:** type text, choose a saved voice, set delivery instructions/seed/steps, synthesize and play WAV audio.

The API process starts automatically when ComfyUI loads the extension. The **GPU model** is only loaded on the first speech/design request.

If the sidebar doesn't appear after restart, hard-refresh the ComfyUI browser tab (`Ctrl+Shift+R`). Confirm the extension is enabled and ComfyUI's startup log has no TTS API errors.

### Local API

The managed API listens on `http://127.0.0.1:8025` by default:

```bash
curl -fsS http://127.0.0.1:8025/health
curl -fsS http://127.0.0.1:8025/v1/audio/voices
curl -fS http://127.0.0.1:8025/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","voice":"default","input":"Hello from the local speech server."}' \
  --output example.mp3
```

If `TTS_API_KEY` is configured, authenticated requests require the header `Authorization: Bearer YOUR_KEY`. The sidebar uses a same-origin gateway so it can call the local service without exposing this key to browser scripts.

### SillyTavern

Set the TTS provider to **OpenAI Compatible**, model `voxcpm2`, voice `default` or a saved ID, and the speech endpoint:

`http://127.0.0.1:8025/v1/audio/speech`

Some SillyTavern versions instead ask for an API root; use `http://127.0.0.1:8025/v1` in that case. Enter saved voice IDs manually if your version does not discover them. When SillyTavern runs on a different device, use the ComfyUI host's LAN/VPN address and enable authenticated LAN serving as described below.

## Voice management API

Clone a voice from a reference file (only with authorization):

```bash
curl -fsS http://127.0.0.1:8025/api/voices/clone \
  -F 'name=Character' \
  -F 'consent=true' \
  -F 'reference_text=Hello, this is my voice.' \
  -F 'audio=@reference.wav'
```

Design a voice:

```bash
curl -fsS http://127.0.0.1:8025/api/voices/design \
  -H 'Content-Type: application/json' \
  -d '{"name":"Narrator","description":"Deep, calm voice with gentle pacing","sample_text":"Welcome to the story.","seed":42}'
```

These return persistent `voice_...` identifiers. To list voices, call `GET /api/voices`; to delete, call `DELETE /api/voices/{voice_id}`.

Sample speech request:

```json
{
  "model": "voxcpm2",
  "input": "Hello world.",
  "voice": "default",
  "response_format": "mp3",
  "speed": 1.0,
  "instructions": "Warm and confident.",
  "seed": 42,
  "steps": 10,
  "cfg": 2.0
}
```

The server buffers completed speech audio before responding. True audio streaming/low first-packet latency are future work.

## GPU coordination and configuration

`ComfyUI-TTS-API/.env` is created by `install.sh` if absent.

```ini
TTS_HOST=127.0.0.1
TTS_PORT=8025
TTS_MODEL=openbmb/VoxCPM2
TTS_AUTO_START=true
TTS_IDLE_UNLOAD_SECONDS=60
TTS_UNLOAD_AFTER_REQUEST=false
TTS_API_KEY=
```

- **ComfyUI owns the process:** the service starts with ComfyUI and shuts down with it.
- **Memory is managed across processes:** the TTS service runs in its own Python/CUDA process. ComfyUI's native model manager does not own its allocations; the extension coordinates unload requests and, if necessary, stops its own service process to free VRAM.
- **Model stays cached for short gaps:** 60 seconds idle by default. Choose a shorter interval for less VRAM pressure or `0` to disable idle unloading (workflow preemption still applies).
- **Workflow priority:** TTS may be interrupted while ComfyUI generates images/videos. A simultaneous SillyTavern request may receive a 503 or be interrupted; perfect two-way scheduling is not available yet.
- **LAN:** To accept direct SillyTavern connections from another device, set `TTS_HOST=0.0.0.0` and a strong `TTS_API_KEY` in `.env`. Restart ComfyUI. LAN bind without an API key is refused. Restrict access using a firewall and preferably a VPN. Do not expose the server directly to the public internet.
- **Independent standalone mode:** `bash start.sh` is available if you intentionally run the speech API *without* ComfyUI. Never start a second copy on the same port.

The sidebar uses ComfyUI's own HTTP security boundary. Anyone allowed to access your ComfyUI instance may be able to control the TTS API via the sidebar routes. Do not expose an unauthenticated ComfyUI instance to untrusted networks.

### Troubleshooting

```bash
curl http://127.0.0.1:8025/health
tail -n 100 ~/ComfyUI/custom_nodes/ComfyUI-TTS-API/data/managed-server.log
```

The first speech request downloads model weights; failures involving Torch/CUDA must be diagnosed in the TTS-specific virtual environment. To disable automatic process startup, set `TTS_AUTO_START=false` and restart ComfyUI.

## Development and validation

```bash
.venv/bin/python -m pip install pytest httpx
.venv/bin/python -m pytest -q
```

GitHub CI covers Python syntax, mocked speech API calls, model pause and idle-unload contracts, and browser script syntax. It does **not** run full GPU inference. The sidebar is registered using ComfyUI's supported `registerSidebarTab` extension API with `WEB_DIRECTORY`. The project intentionally declares empty node mappings.

**Planned:** accelerated inference (vLLM-Omni/Nano-vLLM), Qwen3-TTS backend, cached voice conditioning, audio chunk streaming, and better GPU arbitration with long ComfyUI workloads.
