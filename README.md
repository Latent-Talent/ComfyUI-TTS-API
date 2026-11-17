# ComfyUI TTS API

A local, OpenAI-compatible text-to-speech API for ComfyUI and clients such as SillyTavern. Manage models, saved character voices, and speech generation from the native **TTS API** sidebar.

**Backends:** VoxCPM2, Qwen3-TTS, and IndexTTS 2.5. The service uses isolated Python environments and shares ComfyUI's execution queue without changing ComfyUI's Python, Torch, or CUDA installation.

> Version 0.6. The automated test suite covers API routing, model management, and queue behavior. Inference with every backend has not yet been verified on all GPUs. Audio is returned after complete synthesis; true streaming is not yet implemented.

## Installation

Requirements: an existing ComfyUI installation on Linux/WSL, Python 3.12 for the core TTS server, Git, and FFmpeg for transcoded audio. An NVIDIA GPU is recommended.

First-time install:

~~~bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/Latent-Talent/ComfyUI-TTS-API.git
cd ComfyUI-TTS-API
bash install.sh
~~~

To update an existing installation:

~~~bash
cd ~/ComfyUI/custom_nodes/ComfyUI-TTS-API
git pull --ff-only
~~~

Restart ComfyUI and open **TTS API** in its sidebar. Refresh the browser with Ctrl+Shift+R if the extension does not appear.

The core installer creates `ComfyUI-TTS-API/.venv`. VoxCPM2 is included and is the default TTS backend. Other backends are optional.

## Available backends

| Model ID | Capabilities | Installation |
| --- | --- | --- |
| `voxcpm2` | Speech, voice cloning, voice descriptions, delivery conditioning | Core install |
| `qwen3-tts-1.7b-customvoice` | Nine preset speakers, delivery instructions | `install-qwen.sh` |
| `qwen3-tts-0.6b-customvoice` | Smaller preset-speaker model | `install-qwen.sh` |
| `qwen3-tts-1.7b-voicedesign` | Create a voice from a written description | `install-qwen.sh` |
| `qwen3-tts-1.7b-base` / `qwen3-tts-0.6b-base` | Clone speech from saved reference audio | `install-qwen.sh` |
| `indextts-2.5` | Reference-audio voice cloning and emotion controls | `install-index.sh` |

**Model selection:** Use **Server → Speech model** to choose a backend and **Make API default** to set the model used by aliases such as `tts-1`. The **Download model weights** button is shown only when the selected Qwen or IndexTTS checkpoint is not fully downloaded; it disappears when the local weights are present. The **Test** tab can select another backend without changing the API default. Models load on demand.

### Per-model defaults

In **Server → Model-specific defaults**, selecting any model shows only settings supported by that checkpoint. Each model saves its own profile in `data/model_defaults.json` (file permissions 0600), so changing the default model does not overwrite another model's settings.

- **VoxCPM2:** voice, audio format, playback speed, seed, delivery instructions, inference steps, and CFG.
- **Qwen3-TTS:** voice, audio format, playback speed, seed, and language; delivery instructions are exposed only on checkpoints that support them.
- **IndexTTS 2.5:** voice, audio format, speed, seed, language, emotion source, emotion description, eight-value vector, emotion strength, emotion-reference voice, and duration factor.

The Server tab shows **context-sensitive inputs**: selecting IndexTTS emotion mode `text` reveals an emotion description, `vector` reveals eight sliders/values, `reference` reveals an emotion audio selector, while `neutral` hides all emotion-specific inputs. Audio-reference fields offer a **saved recording selector and a native local audio file picker** rather than a manually typed path. Selecting a new file requires confirmation of permission to use that recording. Clicking **Save defaults** uploads it through the existing Voice Clone API and saves its registered voice ID; the model profile does not contain raw audio or local file paths. The Test tab also accepts local character and emotion reference audio files.

Click **Save defaults** to persist that model's settings, or **Restore model defaults** to reset just that model to its original values. Defaults take effect on new API requests immediately, without restarting ComfyUI or the TTS worker. **Explicit request fields override saved defaults**, including values like `seed: null` or an empty `instructions` string. The Test tab loads saved defaults when its selected model changes; it always requests WAV audio for local playback.

### Qwen3-TTS

~~~bash
cd ~/ComfyUI/custom_nodes/ComfyUI-TTS-API
bash install-qwen.sh
~~~

This installs the Qwen package in the isolated core TTS server environment, not ComfyUI's environment. Restart the managed TTS API after installation.

Preset speaker names: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, and Sohee. When a saved voice recording is selected, Qwen uses its appropriate **Base** checkpoint for cloning.

### IndexTTS 2.5

~~~bash
cd ~/ComfyUI/custom_nodes/ComfyUI-TTS-API
bash install-index.sh
~~~

IndexTTS runs with Python 3.11 in its **own** environment. The installer uses an existing `uv` executable or installs a private copy inside `ComfyUI/models/TTS/IndexTTS-2.5/tools`. It then installs the required Python 3.11 interpreter and IndexTTS packages separately. It does not edit system Python, shell startup files, or either existing ComfyUI virtual environment.

Restart ComfyUI, select **IndexTTS 2.5** in the Server tab, and click **Download model weights**.

IndexTTS requires a **saved reference recording**. Use **Voices → Clone** to register one; selecting `default` without a saved recording does not provide the reference it requires.

**IndexTTS emotion modes:**

| Mode | Input |
| --- | --- |
| Neutral | No extra emotion guidance |
| Auto | Built-in QwenEmotion analysis of the speech text |
| Description | Delivery instructions as an emotion description |
| Manual vector | Eight values ordered happy, angry, sad, afraid, disgusted, melancholic, surprised, calm |
| Reference audio | A second saved voice recording for emotional conditioning |

The Test tab also exposes emotion strength (0–1) and duration factor (0.5–2.0). Supported languages in this integration are English, Chinese, Japanese, Spanish, and Arabic. `Auto` currently defaults to English.

Example API body using a saved voice ID:

~~~json
{
  "model": "indextts-2.5",
  "voice": "voice_YOUR_SAVED_ID",
  "input": "I can't believe you did that!",
  "response_format": "wav",
  "language": "English",
  "emotion_mode": "vector",
  "emotion_vector": [0, 0.8, 0, 0, 0, 0, 0.2, 0],
  "emotion_strength": 0.75,
  "duration_factor": 1.0
}
~~~

## Model storage

Model files are stored under `~/ComfyUI/models/TTS/` by default:

~~~text
ComfyUI/models/TTS/
├── huggingface/hub/             # VoxCPM2 / Qwen3-TTS checkpoints
├── IndexTTS-2.5/
│   ├── checkpoints/            # IndexTTS model weights and auxiliary assets
│   ├── source/                 # Official source and Python 3.11 environment
│   └── tools/                  # Private uv executable, if installed
└── modelscope/                  # Optional ModelScope cache
~~~

Set `TTS_MODELS_DIR=/absolute/path` in the extension's `.env` to use a different TTS model root. This setting is applied to the TTS subprocess only, not the main ComfyUI process.

Saved voices and service logs remain in the extension's data directory (configured by `TTS_DATA_DIR`), independently of model weights.

## Using the TTS API

The managed API starts with ComfyUI and listens on `http://127.0.0.1:8025` by default.

~~~bash
# Check the server
curl -fsS http://127.0.0.1:8025/health

# Get saved voices
curl -fsS http://127.0.0.1:8025/v1/audio/voices

# Generate speech
curl -fS http://127.0.0.1:8025/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","voice":"default","input":"Hello from my local TTS server."}' \
  --output speech.mp3
~~~

If an API key is configured, add `Authorization: Bearer YOUR_KEY` to direct requests.

`/v1/audio/speech` supports MP3 (default), WAV, Opus, AAC, FLAC, and PCM. Backend-specific settings include `instructions`, `language`, and `seed`; `steps` and `cfg` are VoxCPM2 controls.

### SillyTavern

Choose the **OpenAI Compatible** TTS provider. Use the speech endpoint:

`http://127.0.0.1:8025/v1/audio/speech`

If SillyTavern asks for the API root, use `http://127.0.0.1:8025/v1` instead.

Choose a specific model ID or the compatibility alias `tts-1`, which uses your configured local default. Other accepted OpenAI-style aliases include `tts-1-hd` and `gpt-4o-mini-tts`; they do not send requests to OpenAI.

To assign character voices, enter saved voice IDs or unique saved voice names in SillyTavern's **Available Voices** field. Unknown names return an error rather than silently selecting the default. OpenAI voice aliases such as `alloy` and `nova` can map to the local default or an available preset, depending on the backend.

### Voice management

| Endpoint | Function |
| --- | --- |
| `GET /api/voices` | List saved voices |
| `POST /api/voices/clone` | Upload and save a reference recording (multipart form with consent) |
| `POST /api/voices/design` | Create a saved voice from a description with VoxCPM2 or Qwen VoiceDesign |
| `DELETE /api/voices/{voice_id}` | Delete a saved voice |
| `GET /v1/models` | List supported model IDs |

For example, clone a reference recording:

~~~bash
curl -fS http://127.0.0.1:8025/api/voices/clone \
  -F 'name=Character' \
  -F 'consent=true' \
  -F 'reference_text=Hello, this is my voice.' \
  -F 'audio=@reference.wav'
~~~

Use only recordings you have permission to clone.

## ComfyUI queue and GPU management

Each managed speech or voice-design request reserves an actual ComfyUI queue item. Earlier workflows run first; later workflows wait for TTS synthesis and audio encoding to finish.

Completed history entries identify the job as **TTS · Speech completed** or **TTS · Voice design completed**. The internal node carries an opaque ticket, not the spoken text or API credentials. Cancellation, failure, and worker exit release the reservation.

The TTS API process remains online when idle. Model weights are loaded when requested and automatically released after 60 seconds idle by default (`TTS_IDLE_UNLOAD_SECONDS`). ComfyUI may request cached model memory to be released for another workflow. The separate TTS process uses a native CUDA allocator to avoid VoxCPM2 allocator incompatibilities.

**Memory recovery:** A busy or temporarily slow memory-reclaim request no longer causes an immediate worker restart. Reclaim retries are deferred; the worker is restarted only after repeated failed attempts and an unsuccessful health check. Incomplete VoxCPM2/Qwen model loads and inference out-of-memory failures trigger GPU allocator cleanup. When `TTS_UNLOAD_AFTER_REQUEST=true`, two-pass VoxCPM voice design/delivery keeps weights resident until the entire operation is complete. `GET /health` now includes `cuda_memory` with PyTorch allocated/reserved bytes per GPU for the API process, or an unavailable reason when CUDA hasn't been initialized. IndexTTS owns a separate Python/CUDA process, so its memory is identified separately as `index_memory` rather than included in those figures.

**ComfyUI Manager restart:** Manager may restart ComfyUI by replacing the running Python process without changing its PID. The managed TTS worker watches a dedicated parent-liveness pipe (which closes on process replacement), so it shuts down even on this kind of restart. A single-worker lock prevents a second TTS worker from launching while the old one exits. The new ComfyUI instance retries startup when the old worker is gone; only one managed API is intended to run at a time. Normal ComfyUI restarts therefore start a *replacement* API process, not another simultaneous copy.

The standalone `bash start.sh` command runs without ComfyUI queue coordination. Do not run the standalone and managed service simultaneously on the same port.

## Configuration and security

Settings are stored in `ComfyUI-TTS-API/.env`, initially copied from `config.example.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TTS_HOST` | `127.0.0.1` | API bind address |
| `TTS_PORT` | `8025` | API port |
| `TTS_ACTIVE_MODEL` | `voxcpm2` | Default model for alias requests |
| `TTS_MODELS_DIR` | `ComfyUI/models/TTS` | TTS model root |
| `TTS_DATA_DIR` | `./data` | Saved voices and logs |
| `TTS_IDLE_UNLOAD_SECONDS` | `60` | Idle model-unload interval; `0` disables |
| `TTS_UNLOAD_AFTER_REQUEST` | `false` | Unload immediately after every request |
| `TTS_AUTO_START` | `true` | Start managed API with ComfyUI |
| `TTS_API_KEY` | Empty | Key for authenticated direct requests |

Generate or configure an API key from **Server → API key**. The extension writes it to `.env`; an externally supplied environment variable takes precedence.

For remote SillyTavern clients, set `TTS_HOST=0.0.0.0` and a strong `TTS_API_KEY`, then restart ComfyUI. Protect the port with a firewall or VPN. Do not expose an unauthenticated ComfyUI server or TTS API to the public internet. The sidebar's same-origin gateway follows the ComfyUI server's own access boundary.

## Logging and troubleshooting

Operational TTS events are printed in the **main ComfyUI terminal**, with an `[TTS API]` prefix. They cover request timing, model loads, GPU unloads, and failures without including text, voice instructions, reference transcripts, paths, or API keys.

The sanitized event stream is stored in `data/operations.log`. Raw third-party output remains in `data/managed-server.log` and `data/index-worker.log`; **raw logs can include speech text or model diagnostics** and should be treated as sensitive.

~~~bash
curl -fsS http://127.0.0.1:8025/health
tail -n 100 ~/ComfyUI/custom_nodes/ComfyUI-TTS-API/data/managed-server.log
~~~

If IndexTTS fails to start, verify that `install-index.sh` completed and inspect `data/index-worker.log`. If the sidebar is unavailable, confirm the extension loaded during ComfyUI startup and hard-refresh the browser. For general CUDA errors, inspect the *TTS service's* environment rather than changing ComfyUI's Python packages.

## Development

~~~bash
cd ~/ComfyUI/custom_nodes/ComfyUI-TTS-API
.venv/bin/python -m pip install pytest httpx
.venv/bin/python -m pytest -q
~~~

GitHub Actions runs unit tests and JavaScript syntax checks. These do not substitute for end-to-end GPU testing. The extension registers one internal queue-reservation node; speech synthesis is exposed through the sidebar and HTTP API, not user-facing ComfyUI workflow TTS nodes.
