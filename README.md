# ComfyUI TTS API

A **sidebar-managed, local text-to-speech API server** for ComfyUI, with an OpenAI-compatible endpoint for SillyTavern and other clients. Supports **VoxCPM2 and Qwen3-TTS** voice design, cloning and named speakers.

**v0.5 — VoxCPM2 + Qwen3-TTS; API + sidebar with an internal queue reservation node.** ComfyUI starts the service and manages its GPU memory; voice generation is accessed through the sidebar or HTTP API. This project does not filter, censor, rewrite, or send user text to a moderation service. Model output fidelity is not guaranteed by removing application-side filters.

## Highlights

- Dedicated **TTS API** tab in the native ComfyUI sidebar: status, model memory, restart/unload, character voice library, voice design and clone, and a built-in speech test/audio player.
- A managed local speech API starts automatically when ComfyUI starts, and exits when ComfyUI exits. **No separate `start.sh` needed.**
- VoxCPM2 and Qwen3-TTS models load lazily, stay resident between requests, and unload after 60 seconds idle (configurable). Switching model families releases the previous model first.
- **One real ComfyUI queue spot per managed speech or voice-design request:** a small internal output node reserves the queue in FIFO order, waits while the separate TTS worker generates audio, and releases the slot after the API response is prepared. Later ComfyUI workflows wait behind active TTS; TTS waits behind earlier ComfyUI workflows. Only a random ticket is stored in ComfyUI's queue, never text or instructions.
- ComfyUI-specific sidebar API requests are proxied through the ComfyUI HTTP server; browser clients do not need access to the internal API key or a separate TTS URL.
- The standalone OpenAI-style speech endpoint provides MP3 by default, WAV, Opus/Ogg, AAC, FLAC, and 24 kHz signed 16-bit PCM.
- Save cloned voices from a reference recording, or design a voice from a natural-language description.
- The TTS runtime uses an **isolated Python virtual environment**, leaving the existing ComfyUI Python/Torch/CUDA packages untouched.
- No cloud account required. Only model weights are downloaded on first use.

**Important:** The engines use the official native Python packages. Qwen3-TTS GPU inference still needs machine-level verification on the target RTX 4090. vLLM-Omni/Nano-vLLM acceleration and true streaming audio are not implemented.

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

- **Server:** online/offline, model VRAM state, allocator backend, idle unload interval, queue state, API key setup, generation performance, API URL, unload model and restart API.
- **Voices:** list/copy/delete saved voice IDs, create a voice by description, or upload reference audio for cloning.
- **Test:** type text, choose a saved voice, set delivery instructions/seed/steps, synthesize and play WAV audio.

The API process starts automatically when ComfyUI loads the extension. The **GPU model** is only loaded on the first speech/design request.

If the sidebar doesn't appear after restart, hard-refresh the ComfyUI browser tab (`Ctrl+Shift+R`). Confirm the extension is enabled and ComfyUI's startup log has no TTS API errors.

### Install and use Qwen3-TTS (optional)

The existing **VoxCPM2 installation stays intact and is still the default**. To enable Qwen's official Python backend, install it once into the *isolated TTS API* virtual environment:

```bash
cd ~/ComfyUI/custom_nodes/ComfyUI-TTS-API
git pull --ff-only
bash install-qwen.sh
```

This does **not** alter your system Python or ComfyUI's `~/ComfyUI/.venv`. It installs `qwen-tts` and its dependencies only in `ComfyUI-TTS-API/.venv`. It may update shared dependencies used by VoxCPM2 within that isolated TTS environment; the installer runs `pip check` and verifies importability. It does not download model weights during install.

Restart ComfyUI once after updating the custom node, and use the **TTS API → Server → Speech model** controls:

1. Select a supported Qwen model.
2. Optionally click **Download Qwen weights** to cache weights in the normal Hugging Face cache (no GPU load). The sidebar shows download state; the worker log contains detailed transfer progress. Alternatively, the first generation auto-downloads missing model weights.
3. Click **Make API default** if you want SillyTavern's `tts-1` requests to use that model. This saves `TTS_ACTIVE_MODEL` in `.env` and restarts **only the TTS service**, preserving ComfyUI and the API key.
4. Under **Test**, select the backend and use a speaker or saved voice. The selected backend can be tested without changing the global default.

Supported Qwen checkpoints:

| Backend model ID | Features |
| --- | --- |
| `qwen3-tts-1.7b-customvoice` | 9 named speakers; native instructions and multilingual synthesis |
| `qwen3-tts-0.6b-customvoice` | 9 named speakers; no native instruction control in the 0.6B model |
| `qwen3-tts-1.7b-voicedesign` | Native voice descriptions and delivery instructions, spoken text separate |
| `qwen3-tts-1.7b-base` | Cloning with a saved reference recording and optional transcript |
| `qwen3-tts-0.6b-base` | Smaller cloning checkpoint with a saved reference |
 
**Qwen built-in speakers:** Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, and Sohee. The default OpenAI voice names such as `alloy` map to Ryan when Qwen CustomVoice is selected; distinct characters can use these actual named speakers.

**Saved voices work across engines:** Voice Design with Qwen uses its dedicated VoiceDesign model and stores the generated sample as an ordinary persistent `voice_...` recording. When you use that saved voice with Qwen CustomVoice, the server automatically switches to the matching **Base** model for cloning (with a 0.6B Base when a 0.6B model was selected). If the active model is Base and no saved voice is chosen, the API returns an explicit 422 instead of inventing a reference.

Qwen model weights are saved in the default Hugging Face cache, normally `~/.cache/huggingface/hub`. The TTS server keeps at most one active inference model in GPU memory. Downloading a Qwen checkpoint does not preload it in VRAM; selecting a different checkpoint releases the old one before loading the replacement.

**Current limitation:** The Qwen3-TTS Python backend generates full audio before responding. A cold model switch or first-time model download will add latency. Native Qwen voice design and instruction control are distinct from the VoxCPM2 two-pass approximation.

### API key setup from the sidebar

In **TTS API → Server → API key**, choose **Generate & save** to create a secure key, or enter your own (24–256 safe ASCII characters) and click **Save key**. Use **Show/Hide** and **Copy** to obtain the currently entered/generated key for SillyTavern.

The key is persisted in the local `.env` file using restrictive filesystem permissions. Changing it restarts only the managed TTS API, not ComfyUI. Existing keys are never returned to the browser on load, and **a generated key is available only immediately after generation**. Copy it before closing the sidebar.

The Server tab shows whether API authentication is enabled. If `TTS_API_KEY` was supplied as an environment variable to ComfyUI, the sidebar cannot change it: that environment variable takes precedence over `.env`.

If you configure a key, use that same value as SillyTavern's API key for direct connections. ComfyUI's sidebar forwards requests internally and does not need a second key field.

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

Set the TTS provider to **OpenAI Compatible**, model `voxcpm2` (recommended) or SillyTavern's default `tts-1` (accepted compatibility alias), voice `default` or a saved ID, and the speech endpoint:

`http://127.0.0.1:8025/v1/audio/speech`

**Compatibility aliases:** The model IDs `tts-1`, `tts-1-hd`, `tts-1-1106`, `tts-1-hd-1106`, and `gpt-4o-mini-tts` invoke whichever backend is configured in `TTS_ACTIVE_MODEL` (VoxCPM2 by default). They do *not* run an OpenAI model or enable different quality modes. OpenAI default speaker names (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, etc.) map to the single local default voice; for distinct character voices, create voices in the sidebar and use their persistent `voice_...` IDs. Unknown model names still return 422, and unknown custom voice IDs return 404.

### Saved voices in SillyTavern

SillyTavern's **Available Voices** field is a comma-separated list it sends directly to the TTS server. The server accepts built-in aliases regardless of capitalization or surrounding whitespace (for example ` Alloy `, `NOVA`), and it also accepts **saved voice IDs or unique saved voice display names** from ComfyUI → TTS API → Voices. To assign distinct voices to characters, add the saved IDs or names to SillyTavern's Available Voices field, then set the character-to-voice mappings.

You can inspect saved voices using the ComfyUI TTS API sidebar, or `GET /v1/audio/voices`. Unknown voice names return an explicit 404 identifying the requested name; duplicate saved names return 422 and list the IDs to use instead. The server will **never silently substitute** an unknown character's requested voice with the default.

If the API still responds with exactly `{"detail":"'Unknown voice ID'"}`, you are running an older TTS process. Pull the latest repository and fully restart ComfyUI, then check `http://127.0.0.1:8025/health` for `"version": "0.4.1"`.

Some SillyTavern versions instead ask for an API root; use `http://127.0.0.1:8025/v1` in that case. Enter saved voice IDs manually if your version does not discover them. When SillyTavern runs on a different device, use the ComfyUI host's LAN/VPN address and enable authenticated LAN serving as described below.

## Voice description vs. delivery instructions

**Voice Design → Description** defines a reusable voice identity. The only text intentionally synthesized into the saved voice sample is the **Sample text** field. VoxCPM2 only supports inline parenthesized voice instructions, which some model versions pronounce. The implementation therefore creates a short disposable conditioning clip with the description, then clones that timbre into a clean sample containing only the specified words. Creating a new voice requires two synthesis passes.

**Test → Delivery instructions** and the OpenAI-compatible `instructions` parameter control how a particular reply is delivered, without adding instruction words to the final synthesis text. With this VoxCPM2 backend, a short style-conditioning clip is synthesized once for each distinct combination of instruction, voice, seed, steps, and CFG; later requests with identical settings reuse that clip from the worker's bounded in-memory cache. Ordinary speech with no delivery instructions stays single-pass.

Delivery conditioning and cloning fidelity are **best effort**: VoxCPM2 does not provide an independent nonverbal style-parameter API. The voice may not always follow an instruction precisely; the final speech call receives only the intended sample/reply text, but acoustic fidelity still depends on the model. The first use of a style is slower than subsequent cached uses, and cached references are lost when the TTS worker restarts.

## ComfyUI queue reservations

When TTS runs under the managed ComfyUI installation, **each speech or voice-design API request creates an actual ComfyUI queue item** named *TTS · Speech* or *TTS · Voice design*. It is queued at the same priority as ordinary workflows, becomes a running queue item when its turn comes, and blocks following image/video workflows until TTS is finished.

The custom node `LatentTalentTTSQueueSlot` is an internal implementation detail, not a user workflow node: it carries only a random UUID ticket. Speech content, voice descriptions, instructions, and API keys never enter ComfyUI's prompt queue or history. The TTS worker stays in its isolated Python/CUDA process.

**Completed queue history entries are labeled** `TTS · Speech completed` or `TTS · Voice design completed` instead of a generic job number. The internal reservation emits a tiny inline text-only preview containing this label; no speech content or generated audio is written into ComfyUI output files. Clicking View shows only the same completion label. Failed TTS calls are reported as failed ComfyUI queue jobs, not mislabeled as successfully completed.

- New speech requests wait for earlier ComfyUI workflows, and multiple TTS requests get their own queue positions.
- While a TTS reservation runs, later submitted ComfyUI jobs stay pending; they no longer interrupt an active TTS job merely by being submitted.
- A canceled pending reservation is discarded. Interrupting a running reservation stops the managed TTS worker before releasing GPU ownership.
- A crashed worker, lost reservation, or 30-minute watchdog prevents permanent queue blockage. If the queue bridge cannot be reached, managed speech fails explicitly rather than silently bypassing scheduling.
- **Standalone mode** (`bash start.sh`) remains available without ComfyUI queue coordination.

Verify by opening ComfyUI's Queue panel while making a longer TTS request, or inspect `GET /queue` on the ComfyUI server. The queue entry contains no text, only the internal ticket and an operation title. The ComfyUI terminal also logs reservation start/release events without speech content.

This is an internal queue node that registers with ComfyUI only so the queue can hold a real execution slot. It is not meant to be placed in ordinary user workflows.

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

The server buffers completed speech audio before responding. For Qwen, `language` can be set to `Auto`, `English`, `Chinese`, and other supported languages. `seed` is applied to the worker's Torch RNG; `steps` and `cfg` are VoxCPM-only parameters. True audio streaming/low first-packet latency are future work.

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
- **Allocator compatibility:** ComfyUI may use `cudaMallocAsync`, but some Torch CUDA-graph code called by VoxCPM2 requires `checkPoolLiveAllocations` which `cudaMallocAsync` cannot support. Managed and standalone TTS subprocesses therefore explicitly use `PYTORCH_ALLOC_CONF=backend:native` (and its legacy alias). This does not change the CUDA allocator in the ComfyUI process.
- **Memory is managed across processes:** the TTS service runs in its own Python/CUDA process. ComfyUI's native model manager does not own its allocations; the extension coordinates unload requests and, if necessary, stops its own service process to free VRAM.
- **Model stays cached for short gaps:** 60 seconds idle by default. Choose a shorter interval for less VRAM pressure or `0` to disable idle unloading (workflow queue scheduling still applies).
- **Idle API stability:** ComfyUI keeps the lightweight TTS HTTP process running, even when no TTS model is in memory. Ordinary workflows reclaim cached TTS GPU weights only when necessary; they no longer pause/resume or restart a healthy idle TTS worker. A worker is stopped only for explicit shutdown/configuration changes or when GPU reclaim fails.
- **Queue scheduling:** ComfyUI and managed TTS use the same FIFO execution queue, including requests arriving through the separate SillyTavern API. An active TTS job occupies the queue until synthesis and audio encoding finish. If a workflow is already running, TTS waits for its turn rather than consuming GPU memory concurrently. ComfyUI's manual Interrupt or queue cancellation can abort a queued/running TTS reservation; the managed speech worker is terminated when necessary before allowing the next GPU job to run.
- **LAN:** To accept direct SillyTavern connections from another device, set `TTS_HOST=0.0.0.0` and a strong `TTS_API_KEY` in `.env`. Restart ComfyUI. LAN bind without an API key is refused. Restrict access using a firewall and preferably a VPN. Do not expose the server directly to the public internet.
- **Independent standalone mode:** `bash start.sh` is available if you intentionally run the speech API *without* ComfyUI. Never start a second copy on the same port.

The sidebar uses ComfyUI's own HTTP security boundary. Anyone allowed to access your ComfyUI instance may be able to control the TTS API via the sidebar routes. Do not expose an unauthenticated ComfyUI instance to untrusted networks.

### VoxCPM seed compatibility

Some installed versions of `voxcpm` do not accept the `seed` argument in `VoxCPM._generate()`, even though newer upstream versions do. The TTS adapter detects this automatically. For newer versions, it forwards `seed`; for older versions, it seeds the isolated worker's PyTorch, NumPy, and Python random-number generators without passing an unsupported keyword. The previous `VoxCPM._generate() got an unexpected keyword argument 'seed'` failure in **Voices → Design** and **Test → Generate** should no longer occur. Seed reproducibility on older engines is best effort and may differ across hardware/software versions.

Updating the project requires a `git pull --ff-only` and ComfyUI restart; it does not require changing the installed `voxcpm` package.

### Troubleshooting

```bash
curl http://127.0.0.1:8025/health
tail -n 100 ~/ComfyUI/custom_nodes/ComfyUI-TTS-API/data/managed-server.log
```

The first speech request downloads model weights; failures involving Torch/CUDA must be diagnosed in the TTS-specific virtual environment. If you previously encountered `cudaMallocAsync does not yet support checkPoolLiveAllocations`, pull this update and restart ComfyUI so it launches a fresh TTS child with the native allocator. To disable automatic process startup, set `TTS_AUTO_START=false` and restart ComfyUI.

## TTS console logging (without speech text)

The managed TTS worker now reports operational events **directly in the main ComfyUI terminal/console**. No extra sidebar tab or activity endpoint is needed. Watch the same terminal where ComfyUI started: events are labeled `[TTS API]`. This covers requests, model loading/unloading and switching, Qwen downloads, voice design/cloning, timing, cache usage, and failures.

For example:

```text
[TTS API] request.started req=03a4cd02a883 kind=speech status=starting
[TTS API] generation.started req=03a4cd02a883 model=voxcpm2 backend=voxcpm2 voice_type=default characters=81
[TTS API] model.load.completed model=voxcpm2 backend=voxcpm2 elapsed_ms=1500.3
[TTS API] generation.completed req=03a4cd02a883 model=voxcpm2 audio_seconds=3.8 elapsed_ms=4200.7
[TTS API] request.completed req=03a4cd02a883 kind=speech status=ok elapsed_ms=4203.1
```

These lines are **examples**, not measured timings. The event relay prints *only* fields allowed by a strict schema: it will not print the spoken text, voice descriptions, delivery instructions, transcripts, API keys, file paths, or exception messages. Failures are identified by exception class (e.g., `RuntimeError`) rather than dumping potentially sensitive traceback arguments.

A small rotating `data/operations.log` file is used internally to forward these sanitized events from the separate worker into the ComfyUI console, while raw stdout/stderr from third-party libraries remains in `data/managed-server.log` and is **not mirrored to the console**. Third-party raw output may contain user content.

## Development and validation

```bash
.venv/bin/python -m pip install pytest httpx
.venv/bin/python -m pytest -q
```

GitHub CI covers Python syntax, mocked speech API calls, model pause and idle-unload contracts, and browser script syntax. It does **not** run full GPU inference. The sidebar is registered using ComfyUI's supported `registerSidebarTab` extension API with `WEB_DIRECTORY`. The project registers only an internal TTS queue-reservation output node; it does not expose general-purpose workflow TTS nodes.

**Planned:** accelerated inference (vLLM-Omni/Nano-vLLM), audio chunk streaming, and better GPU arbitration with long ComfyUI workloads.
