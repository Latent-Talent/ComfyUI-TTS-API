# ComfyUI TTS API

Local text-to-speech for **ComfyUI** and **SillyTavern**, with voice cloning and natural-language voice design.

**Status: early implementation (v0.2).** This version uses the official **VoxCPM2 Python API**. ComfyUI **automatically starts and stops** the API in a separate Python virtual environment; the speech model is loaded on demand and relinquishes VRAM when idle or another ComfyUI workflow needs the GPU. There is no application-layer profanity filter, text sanitizer, online moderation request, or required cloud service. A model may still omit or mispronounce some inputs. **Qwen3-TTS, vLLM-Omni acceleration, true token streaming, and multiprocess scheduling are not yet implemented.** This first version prioritizes a functional base and a stable API contract.

## Features

- OpenAI-like `POST /v1/audio/speech`, returning actual MP3 by default (or WAV, AAC, FLAC, Ogg/Opus, raw 24 kHz PCM).
- `GET /v1/models`, `GET /v1/audio/voices`, and `GET /health`.
- Upload a reference recording for a reusable named cloned voice.
- Design a voice by textual description, synthesize a reference sample, and save it.
- Persistent local voice IDs for mapping SillyTavern characters to voice profiles.
- ComfyUI nodes: **TTS Generate (API)**, **TTS Clone Voice (API)**, **TTS Design Voice (API)**.
- Automatic API process start/stop with ComfyUI; no separate server startup command.
- Lazy model loading, idle VRAM unloading (default 60 seconds), and prompt-triggered GPU preemption.
- Explicit `POST /api/model/unload`, and configurable unloading after each request.
- Seed, inference steps, CFG, and per-request delivery instructions.
- FastAPI docs at `http://127.0.0.1:8025/docs`.
- Optional authentication for LAN mode.
- No installation into the existing ComfyUI Python environment.

## Linux / WSL installation

Clone the project **inside** ComfyUI's `custom_nodes` directory for automatic service management and node discovery. The API runs in an isolated subprocess managed by ComfyUI.

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/Latent-Talent/ComfyUI-TTS-API.git
cd ComfyUI-TTS-API
bash install.sh

# Then start or restart ComfyUI normally — no separate TTS server command.
```

Requires an existing Python 3.12 executable, NVIDIA GPU for fast inference, and `ffmpeg` on PATH for MP3 output. The installation script uses Python 3.12 for this project's `.venv`; **it does not alter system Python or ComfyUI's virtual environment**. On WSL the first generation downloads model weights into your normal Hugging Face cache. GPU inference and CUDA/Torch compatibility must be checked on the target host; this code has not been run on your GPU.

If Python 3.12 is installed under a different executable, run `PYTHON=/absolute/path/to/python3.12 bash install.sh`. There is **no need to run `start.sh`** while ComfyUI is running. Restart ComfyUI after installing the custom nodes; it will start the lightweight speech API automatically. Model weights load only on the first speech request.

Copy `config.example.env` to `.env` (install.sh does this if missing) and edit `TTS_MODEL`, `TTS_API_KEY`, `TTS_IDLE_UNLOAD_SECONDS`, etc. `TTS_HOST` defaults to loopback. To serve SillyTavern on another LAN host automatically, set `TTS_HOST=0.0.0.0` and a nonempty `TTS_API_KEY` in `.env` before starting ComfyUI. LAN mode without a key is refused. `bash start.sh --lan` remains an **optional standalone mode** for use when ComfyUI is not running. Restrict the port with your firewall and preferably a VPN. Do not expose this service directly to the public internet.

## SillyTavern

Under **Extensions → TTS → OpenAI Compatible**, set:

- Endpoint/base URL, as required by your SillyTavern version: `http://127.0.0.1:8025/v1/audio/speech` (the speech route), or `http://127.0.0.1:8025/v1` if it requests an API root.
- Model: `voxcpm2`
- Format: `mp3` (default)
- Voice ID: `default` or a `voice_...` ID from `GET /v1/audio/voices`.
- API key: the `TTS_API_KEY` from `.env`, if configured.

If SillyTavern is on another machine, use the server's VPN/LAN IP instead of loopback. SillyTavern versions vary in how they populate voice dropdowns; voice IDs can be entered manually if auto-discovery is unavailable. **ComfyUI must remain running** to provide the automatically managed API. Standalone `start.sh` is available for independent service operation.

### Test it directly

```bash
curl -fsS http://127.0.0.1:8025/health
curl -fsS http://127.0.0.1:8025/v1/audio/voices
curl -fS http://127.0.0.1:8025/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","voice":"default","input":"Hello from the local speech server.","response_format":"mp3"}' \
  --output test.mp3
```

When an API key is configured, append `-H 'Authorization: Bearer YOUR_KEY'` to authenticated requests.

### Clone a voice

```bash
curl -fsS http://127.0.0.1:8025/api/voices/clone \
  -F 'name=Example character' \
  -F 'consent=true' \
  -F 'reference_text=Hello, this is my recording.' \
  -F 'audio=@reference.wav'
```

This returns a saved voice ID. Cloning works best with clear voice-only audio. Provide an accurate reference transcript for the high-fidelity reference-audio prompting mode.

### Design a voice

```bash
curl -fsS http://127.0.0.1:8025/api/voices/design \
  -H 'Content-Type: application/json' \
  -d '{"name":"Narrator","description":"Deep, calm adult voice with gentle pacing","sample_text":"Welcome to my story.","seed":42}'
```

Design creates a reference sample and stores it in the local voice library. The resulting voice ID can be used with SillyTavern or the ComfyUI Generate node.

## API contract

`POST /v1/audio/speech` body:

```json
{
  "model": "voxcpm2",
  "input": "Hello world",
  "voice": "default",
  "response_format": "mp3",
  "speed": 1.0,
  "instructions": "Calm and confident",
  "seed": 42,
  "steps": 10,
  "cfg": 2.0
}
```

Supported formats: `mp3`, `wav`, `opus` (Ogg container), `aac`, `flac`, `pcm` (24 kHz mono 16-bit LE). Speech generation presently buffers the complete WAV before transcoding and responding; HTTP streaming and sub-second first audio are roadmap work. Requests are serialized at the model level to avoid simultaneous CUDA operations within one process. Running with multiple Uvicorn workers would separately load the model and waste VRAM; **use one worker**.

Voice management endpoints: `POST /api/voices/clone` (multipart), `POST /api/voices/design` (JSON), `GET /api/voices`, `DELETE /api/voices/{voice_id}`.

## GPU memory and lifecycle

- ComfyUI automatically starts the **API process** when the custom node loads. This uses only a small amount of host RAM; **no model is preloaded into VRAM**.
- The first TTS or voice-design request loads VoxCPM2 onto the GPU. Subsequent requests reuse it while active.
- After `TTS_IDLE_UNLOAD_SECONDS=60` seconds with no inference, the API unloads VoxCPM2 and frees its CUDA caching allocator.
- When any non-TTS ComfyUI workflow is submitted, the node asks the API to **pause inference and release VoxCPM VRAM before the prompt enters the ComfyUI queue**. It resumes the TTS API after the ComfyUI queue is idle.
- If the TTS worker is still generating and cannot release VRAM promptly, ComfyUI **terminates its own TTS process**, aborting the active audio response if necessary, and restarts it after the workflow is done. It never terminates an unrelated process already listening on the configured port.
- In mixed workflows containing TTS and image/video nodes, the TTS node temporarily resumes the worker during its own generation and relinquishes it afterward.
- When ComfyUI exits, the owned TTS process is stopped. An orphaned worker also exits after detecting its parent is gone.

The speech backend runs in a separate process to avoid modifying ComfyUI's Torch installation. This means it is **not a native ComfyUI-managed model**; CUDA allocations cannot be freed directly from the ComfyUI process. Memory is coordinated through controlled unloads and process termination. SillyTavern requests arriving when ComfyUI is using most of the GPU may receive a temporary 503 or fail for insufficient VRAM; full two-way memory negotiation is future work.

Set `TTS_AUTO_START=false` if you intentionally want to manage the TTS API independently. Set `TTS_IDLE_UNLOAD_SECONDS=0` to disable idle unloading, although ComfyUI prompt-triggered preemption will still apply to a managed worker.

### Tests

```bash
.venv/bin/python -m pip install pytest httpx
.venv/bin/python -m pytest -q
```

The tests mock inference and do not claim to benchmark GPU execution or acoustical quality.

## Architecture and next steps

```
SillyTavern -- OpenAI speech JSON --> FastAPI --> VoxCPM2 --> MP3/WAV
ComfyUI custom nodes -------------> FastAPI --> persisted character voice references
```

Near-term priorities: Qwen3-TTS adapter; vLLM-Omni/Nano-vLLM accelerator behind identical endpoints; first-audio streaming; per-voice conditioning cache; better two-way GPU memory negotiation; regression tests on an RTX 4090.

### Safety and privacy

All generation and voice data remain local. Upload only recordings you are authorized to use; clone requests require an explicit consent flag. The service has no content-filtering pipeline. For shared use, configure an API key, firewall, VPN, and filesystem permissions. There is no web authentication UI or upload quota per user in v0.1.
