# ComfyUI TTS API

Local text-to-speech for **ComfyUI** and **SillyTavern**, with voice cloning and natural-language voice design.

**Status: first implementation (v0.1).** This version uses the official **VoxCPM2 Python API**. The speech model runs in a standalone Python virtual environment; ComfyUI communicates with it over HTTP. There is no application-layer profanity filter, text sanitizer, online moderation request, or required cloud service. A model may still omit or mispronounce some inputs. **Qwen3-TTS, vLLM-Omni acceleration, true token streaming, and multiprocess scheduling are not yet implemented.** This first version prioritizes a functional base and a stable API contract.

## Features

- OpenAI-like `POST /v1/audio/speech`, returning actual MP3 by default (or WAV, AAC, FLAC, Ogg/Opus, raw 24 kHz PCM).
- `GET /v1/models`, `GET /v1/audio/voices`, and `GET /health`.
- Upload a reference recording for a reusable named cloned voice.
- Design a voice by textual description, synthesize a reference sample, and save it.
- Persistent local voice IDs for mapping SillyTavern characters to voice profiles.
- ComfyUI nodes: **TTS Generate (API)**, **TTS Clone Voice (API)**, **TTS Design Voice (API)**.
- Lazy model loading and explicit `POST /api/model/unload`. Configurable unloading after each request.
- Seed, inference steps, CFG, and per-request delivery instructions.
- FastAPI docs at `http://127.0.0.1:8025/docs`.
- Optional authentication for LAN mode.
- No installation into the existing ComfyUI Python environment.

## Linux / WSL installation

This project must be cloned **inside** ComfyUI's `custom_nodes` directory for ComfyUI node discovery. The API process runs separately.

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/Latent-Talent/ComfyUI-TTS-API.git
cd ComfyUI-TTS-API
bash install.sh
bash start.sh
```

Requires an existing Python 3.12 executable, NVIDIA GPU for fast inference, and `ffmpeg` on PATH for MP3 output. The installation script uses Python 3.12 for this project's `.venv`; **it does not alter system Python or ComfyUI's virtual environment**. On WSL the first generation downloads model weights into your normal Hugging Face cache. GPU inference and CUDA/Torch compatibility must be checked on the target host; this code has not been run on your GPU.

If Python 3.12 is installed under a different executable, run `PYTHON=/absolute/path/to/python3.12 bash install.sh`. Start the server separately from ComfyUI. Restart ComfyUI after installing the custom nodes.

Copy `config.example.env` to `.env` (install.sh does this if missing) and edit `TTS_MODEL`, `TTS_API_KEY`, etc. `TTS_HOST` defaults to loopback. Use `bash start.sh --lan` to bind to `0.0.0.0`; LAN mode requires a configured `TTS_API_KEY`. Restrict the port with your firewall and preferably a VPN. Do not expose this service directly to the public internet.

## SillyTavern

Under **Extensions → TTS → OpenAI Compatible**, set:

- Endpoint/base URL, as required by your SillyTavern version: `http://127.0.0.1:8025/v1/audio/speech` (the speech route), or `http://127.0.0.1:8025/v1` if it requests an API root.
- Model: `voxcpm2`
- Format: `mp3` (default)
- Voice ID: `default` or a `voice_...` ID from `GET /v1/audio/voices`.
- API key: the `TTS_API_KEY` from `.env`, if configured.

If SillyTavern is on another machine, use the server's VPN/LAN IP instead of loopback. SillyTavern versions vary in how they populate voice dropdowns; voice IDs can be entered manually if auto-discovery is unavailable.

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

Near-term priorities: Qwen3-TTS adapter; vLLM-Omni/Nano-vLLM accelerator behind identical endpoints; first-audio streaming; per-voice conditioning cache; GPU resource management for simultaneous ComfyUI image/video work; regression tests on an RTX 4090.

### Safety and privacy

All generation and voice data remain local. Upload only recordings you are authorized to use; clone requests require an explicit consent flag. The service has no content-filtering pipeline. For shared use, configure an API key, firewall, VPN, and filesystem permissions. There is no web authentication UI or upload quota per user in v0.1.
