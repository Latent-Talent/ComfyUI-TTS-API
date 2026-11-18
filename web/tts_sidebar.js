import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const ROUTE = "/tts-api";
let activePanel = null;

// Session-only UI state. Do not retain API keys, permission checkboxes,
// local file inputs, or microphone streams. The server instance identifier
// invalidates the snapshot when ComfyUI itself restarts (not the TTS worker).
const SIDEBAR_SESSION_KEY = "latent-talent.tts-api.sidebar-session.v1";
const SESSION_FIELDS = [
    "selected-model", "test-model", "test-language", "test-voice",
    "test-text", "test-delivery", "test-seed", "test-steps",
    "index-emo-mode", "index-emo-reference", "index-emo-strength",
    "index-duration", "voice-search",
    "design-model", "design-name", "design-description",
    "design-sample", "design-seed", "clone-name", "clone-transcript",
];
let savedSidebarSession = null;
let comfyuiSessionId = null;

function storedSidebarSession() {
    try {
        const raw = window.sessionStorage.getItem(SIDEBAR_SESSION_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (_) {
        return null; // Session storage can be disabled by the browser.
    }
}

function clearSidebarSession() {
    savedSidebarSession = null;
    try { window.sessionStorage.removeItem(SIDEBAR_SESSION_KEY); } catch (_) { /* Unavailable */ }
}

function rememberComfyuiSession(instanceId) {
    if (typeof instanceId !== "string" || !instanceId) return false;
    const prior = comfyuiSessionId ||
        (savedSidebarSession || storedSidebarSession())?.comfyuiSession;
    comfyuiSessionId = instanceId;
    if (prior && prior !== instanceId) {
        clearSidebarSession();
        return true;
    }
    return false;
}

async function request(path, options = {}) {
    const response = await api.fetchApi(ROUTE + path, options);
    if (!response.ok) {
        let detail = "HTTP " + response.status;
        try {
            const body = await response.json();
            detail = body.error || body.detail || detail;
            if (typeof detail !== "string") detail = JSON.stringify(detail);
        } catch (_) { /* Preserve HTTP status */ }
        throw new Error(detail);
    }
    return response;
}

function template() {
    return `
    <section class="tts-ui" aria-label="Text to speech API manager">
      <header class="tts-head">
        <div class="tts-head-icon">♫</div>
        <div class="tts-head-text"><strong>TTS API</strong><small>Local voice service</small></div>
        <span class="tts-status-dot" title="Checking server"></span>
      </header>
      <div class="tts-banner" role="status">Connecting to service…</div>
      <nav class="tts-tabs" aria-label="TTS sections">
        <button type="button" class="is-active" data-tab="server">Server</button>
        <button type="button" data-tab="voices">Voices</button>
        <button type="button" data-tab="test">Test</button>
      </nav>
      <div class="tts-panel" data-panel="server">
        <div class="tts-section-title">Actions</div>
        <div class="tts-actions">
          <button type="button" data-action="refresh">Refresh</button>
          <button type="button" data-action="unload">Unload model</button>
          <button type="button" data-action="restart">Restart API</button>
        </div>
        <div class="tts-section-title">Speech model</div>
        <label>Default API backend<select data-selected-model>
          <option value="voxcpm2">VoxCPM2</option>
        </select></label>
        <div class="tts-actions">
          <button type="button" data-action="apply-model">Make API default</button>
          <button type="button" data-action="download-model">Download model weights</button>
        </div>
        <p class="tts-hint" data-model-hint>Models load on demand. Switching the default restarts only the managed TTS API, not ComfyUI.</p>
        <p class="tts-hint" data-download-status>No model download active.</p>
        <div class="tts-section-title">Model-specific defaults</div>
        <p class="tts-hint">Saved separately for each model. They apply when the API request omits a setting; explicit request values take priority.</p>
        <div class="tts-defaults-fields" data-model-defaults-fields>
          <p class="tts-muted">Loading model settings…</p>
        </div>
        <div class="tts-actions">
          <button type="button" class="tts-primary" data-action="save-model-defaults">Save defaults</button>
          <button type="button" data-action="reset-model-defaults">Restore model defaults</button>
        </div>
        <div class="tts-section-title">Connection</div>
        <div class="tts-endpoint">
          <code data-endpoint>http://127.0.0.1:8025/v1/audio/speech</code>
          <button type="button" data-action="copy-url" title="Copy speech API URL">Copy</button>
        </div>
        <p class="tts-hint">Use this URL with SillyTavern’s OpenAI Compatible TTS provider. If SillyTavern is on another device, replace localhost with your ComfyUI server address and enable authenticated LAN access in .env.</p>
        <div class="tts-section-title">API key</div>
        <div class="tts-info"><div><span>Authentication</span><strong data-api-key-status>Checking</strong></div></div>
        <label class="tts-key-label">New or generated API key
          <input data-api-key type="password" value="" autocomplete="off" spellcheck="false" placeholder="Generate a key or enter your own (24+ chars)">
        </label>
        <div class="tts-actions tts-key-actions">
          <button type="button" data-action="generate-key">Generate &amp; save</button>
          <button type="button" data-action="save-key">Save key</button>
          <button type="button" data-action="copy-key">Copy</button>
          <button type="button" data-action="toggle-key">Show</button>
        </div>
        <p class="tts-hint">Existing keys are never displayed. A generated key is shown only now — copy it to SillyTavern. Saving a key immediately restarts the managed API; it does not restart ComfyUI.</p>
        <div class="tts-section-title">Last generation</div>
        <p class="tts-muted" data-bench>No speech generated yet.</p>
        <div class="tts-section-title">Inference service</div>
        <div class="tts-info">
          <div><span>Backend</span><strong data-model>VoxCPM2</strong></div>
          <div><span>API version</span><strong data-version>Unknown</strong></div>
          <div><span>API</span><strong data-service>Checking</strong></div>
          <div><span>Model memory</span><strong data-memory>Not loaded</strong></div>
          <div><span>Execution</span><strong data-runtime>Not loaded</strong></div>
          <div><span>CUDA allocator</span><strong data-allocator>Native</strong></div>
          <div><span>Idle unload</span><strong data-idle>60s</strong></div>
          <div><span>ComfyUI queue</span><strong data-queue>Checking</strong></div>
        </div>
      </div>
      <div class="tts-panel" data-panel="voices" hidden>
        <div class="tts-section-title">Saved character voices</div>
        <label>Search voices
          <input data-voice-search type="search" placeholder="Search name, description or voice ID" autocomplete="off">
        </label>
        <div class="tts-voice-toolbar">
          <span class="tts-muted" data-voice-count role="status" aria-live="polite">Loading voices…</span>
          <button type="button" data-action="refresh-voices">Refresh</button>
        </div>
        <div class="tts-voices" data-voices><p class="tts-muted">Loading…</p></div>
        <details class="tts-accordion">
          <summary>Design voice from description</summary>
          <label>Voice name<input data-design-name type="text" maxlength="120" placeholder="Narrator" value="Narrator"></label>
          <label>Voice design backend<select data-design-model>
            <option value="voxcpm2">VoxCPM2</option>
            <option value="qwen3-tts-1.7b-voicedesign">Qwen3-TTS 1.7B VoiceDesign</option>
          </select></label>
          <label>Description<textarea data-design-description rows="3" placeholder="Deep, calm adult voice with a warm tone"></textarea></label>
          <label>Sample text<textarea data-design-sample rows="2">Hello, this is my new voice.</textarea></label>
          <label>Seed<input data-design-seed type="number" value="42" min="0" max="2147483647"></label>
          <button type="button" class="tts-primary" data-action="design">Create voice</button>
        </details>
        <details class="tts-accordion">
          <summary>Clone voice from recording</summary>
          <label>Name<input data-clone-name type="text" maxlength="120" placeholder="Character name"></label>
          <label>Reference audio<input data-clone-file type="file" accept="audio/*,.wav,.flac,.ogg,.mp3"></label>
          <div class="tts-record-controls">
            <span class="tts-muted">Or record with your microphone</span>
            <div class="tts-actions">
              <button type="button" data-action="clone-record">Record voice</button>
              <button type="button" data-action="clone-discard" hidden>Discard recording</button>
            </div>
            <small class="tts-muted" data-clone-record-status role="status" aria-live="polite"></small>
            <audio data-clone-record-preview controls preload="metadata" hidden aria-label="Recorded voice preview"></audio>
          </div>
          <label>Reference transcript (optional)<textarea data-clone-transcript rows="2"></textarea></label>
          <label class="tts-check"><input data-clone-consent type="checkbox"> I have permission to use this voice.</label>
          <button type="button" class="tts-primary" data-action="clone">Save cloned voice</button>
        </details>
      </div>
      <div class="tts-panel" data-panel="test" hidden>
        <div class="tts-section-title">Test speech generation</div>
        <label>Speech backend<select data-test-model>
          <option value="voxcpm2">VoxCPM2</option>
        </select></label>
        <label>Language<select data-test-language>
          <option value="Auto">Auto detect</option>
          <option value="English">English</option>
          <option value="Chinese">Chinese</option>
          <option value="Japanese">Japanese</option>
          <option value="Korean">Korean</option>
          <option value="German">German</option>
          <option value="French">French</option>
          <option value="Spanish">Spanish</option>
          <option value="Arabic">Arabic</option>
          <option value="Italian">Italian</option>
          <option value="Portuguese">Portuguese</option>
          <option value="Russian">Russian</option>
        </select></label>
        <label>Character voice<select data-test-voice><option value="default">Default voice</option></select></label>
        <label>Text<textarea data-test-text rows="5">Hello! This is a test of my local text to speech server.</textarea></label>
        <label>Delivery instructions (optional)<textarea data-test-delivery rows="2" placeholder="Warm and relaxed"></textarea></label>
        <div class="tts-test-row">
          <label>Seed<input data-test-seed type="number" value="42" min="0" max="2147483647"></label>
          <label>Steps<input data-test-steps type="number" value="10" min="1" max="100"></label>
        </div>

        <details data-index-emotion hidden class="tts-accordion">
          <summary>IndexTTS 2.5 · Emotion and timing</summary>
          <p class="tts-hint">Requires a saved voice recording. Auto and Description use the built-in emotion analyzer.</p>
          <label>Emotion mode<select data-index-emo-mode>
            <option value="neutral">Neutral</option>
            <option value="auto">Automatically analyze speech text</option>
            <option value="text">Use delivery instructions as emotion description</option>
            <option value="vector">Manual 8-value vector</option>
            <option value="reference">Emotion from a second saved voice</option>
          </select></label>
          <label data-index-emo-ref-row hidden>Emotion reference<select data-index-emo-reference><option value="default">Select saved voice</option></select></label>
          <label data-index-emo-file-row hidden class="tts-audio-picker">Or browse for an emotion recording
            <input data-index-emo-file type="file" accept="audio/*,.wav,.flac,.ogg,.mp3,.aiff">
          </label>
          <label data-index-emo-file-consent-row hidden class="tts-check">
            <input data-index-emo-file-consent type="checkbox"> I have permission to use this recording
          </label>
          <div data-index-emo-vector-row hidden>
            <p class="tts-hint">Intensity 0–1 for each emotion.</p>
            <div class="tts-emotion-grid">
              <label>Happy<input type="number" data-emotion="happy" min="0" max="1" step="0.05" value="0"></label>
              <label>Angry<input type="number" data-emotion="angry" min="0" max="1" step="0.05" value="0"></label>
              <label>Sad<input type="number" data-emotion="sad" min="0" max="1" step="0.05" value="0"></label>
              <label>Afraid<input type="number" data-emotion="afraid" min="0" max="1" step="0.05" value="0"></label>
              <label>Disgusted<input type="number" data-emotion="disgusted" min="0" max="1" step="0.05" value="0"></label>
              <label>Melancholic<input type="number" data-emotion="melancholic" min="0" max="1" step="0.05" value="0"></label>
              <label>Surprised<input type="number" data-emotion="surprised" min="0" max="1" step="0.05" value="0"></label>
              <label>Calm<input type="number" data-emotion="calm" min="0" max="1" step="0.05" value="0"></label>
            </div>
          </div>
          <div class="tts-test-row">
            <label>Emotion strength<input data-index-emo-strength type="number" min="0" max="1" step="0.05" value="0.65"></label>
            <label>Duration factor<input data-index-duration type="number" min="0.5" max="2" step="0.05" value="1"></label>
          </div>
        </details>
        <p class="tts-hint" data-test-model-note>Qwen3-TTS has separate CustomVoice, VoiceDesign, and Base checkpoints. Saved voice references use Base automatically.</p>
        <button type="button" class="tts-primary" data-action="generate">Generate speech</button>
        <audio data-audio controls hidden></audio>
      </div>
      <div class="tts-feedback" role="status" aria-live="polite" hidden></div>
    </section>`;
}

// Browser recorders typically produce Opus/AAC, while the voice API accepts WAV.
// Decode and resample locally to mono PCM-16 before using the existing clone API.
async function recordedAudioToWav(blob) {
    const Context = window.AudioContext || window.webkitAudioContext;
    const OfflineContext = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Context || !OfflineContext) {
        throw new Error("This browser cannot convert microphone recordings to WAV.");
    }
    const context = new Context();
    let decoded;
    try {
        decoded = await context.decodeAudioData(await blob.arrayBuffer());
    } finally {
        await context.close();
    }
    if (!Number.isFinite(decoded.duration) || decoded.duration < 0.3 ||
        decoded.duration > 120) {
        throw new Error("Record between 0.3 and 120 seconds of audio.");
    }
    const sampleRate = 24000;
    const frames = Math.ceil(decoded.duration * sampleRate);
    const offline = new OfflineContext(1, frames, sampleRate);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0);
    const wav = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(wav);
    const ascii = (offset, value) => {
        for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
    };
    ascii(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // uncompressed PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    ascii(36, "data");
    view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
        const value = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + 2 * i, value < 0
            ? Math.round(value * 32768) : Math.round(value * 32767), true);
    }
    return new File([wav], "microphone-recording.wav", {type: "audio/wav"});
}

function mount(container) {
    if (activePanel) activePanel.destroy();
    container.innerHTML = template();
    const root = container.querySelector(".tts-ui");
    let timer = null;
    let audioURL = null;
    let lastAudioBlob = null;
    let destroyed = false;
    let initializing = true;
    let sessionVerified = false;
    let skipSessionCapture = false;
    let modelList = [{ id: "voxcpm2", label: "VoxCPM2", kind: "voxcpm" }];
    let modelChoicesInitialized = false;
    let defaultsByModel = {};
    let savedVoices = [];
    const voicePreviewUrls = new Map();
    let renderedDefaultsModel = null;
    let downloadStatus = null;
    let latestCompletedDownload = null;
    const emotionNames = ["happy", "angry", "sad", "afraid", "disgusted",
                          "melancholic", "surprised", "calm"];
    const qwenSpeakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
                          "Ryan", "Aiden", "Ono_Anna", "Sohee"];
    const $ = (selector) => root.querySelector(selector);

    function captureSidebarSession() {
        if (destroyed || initializing || !sessionVerified || !comfyuiSessionId) return;
        const fields = {};
        for (const name of SESSION_FIELDS) {
            const control = $("[data-" + name + "]");
            if (control) fields[name] = control.value;
        }
        const emotions = {};
        root.querySelectorAll("[data-emotion]").forEach((field) => {
            emotions[field.dataset.emotion] = field.value;
        });
        const defaults = {};
        root.querySelectorAll("[data-model-defaults-fields] [data-default-field]").forEach((field) => {
            const vectors = field.querySelectorAll("[data-vector-index]");
            defaults[field.dataset.defaultField] = vectors.length
                ? [...vectors].map((item) => item.value) : field.value;
        });
        const snapshot = {
            comfyuiSession: comfyuiSessionId,
            tab: root.querySelector("[data-tab].is-active")?.dataset.tab || "server",
            fields, emotions, defaults,
            defaultsModel: $("[data-selected-model]").value,
            voiceAccordions: [...root.querySelectorAll('[data-panel="voices"] > details')]
                .map((details) => details.open),
            indexAccordion: $("[data-index-emotion]").open,
            containerScroll: container.scrollTop,
            rootScroll: root.scrollTop,
        };
        // Generated audio is kept in memory only, never in browser storage.
        savedSidebarSession = {...snapshot, lastAudioBlob};
        try { window.sessionStorage.setItem(SIDEBAR_SESSION_KEY, JSON.stringify(snapshot)); }
        catch (_) { /* The in-memory snapshot is still usable. */ }
    }

    const persistOnEdit = () => captureSidebarSession();
    root.addEventListener("input", persistOnEdit);
    root.addEventListener("change", persistOnEdit);
    root.addEventListener("toggle", persistOnEdit, true);
    const onPageHide = () => captureSidebarSession();
    window.addEventListener("pagehide", onPageHide);

    function restoreField(name, value) {
        const control = $("[data-" + name + "]");
        if (!control || value == null) return;
        // Never insert an unavailable backend or deleted voice into a select.
        if (control.tagName === "SELECT" &&
            ![...control.options].some((item) => item.value === String(value))) return;
        control.value = String(value);
    }

    async function restoreSidebarSession(snapshot) {
        const fields = snapshot.fields || {};
        restoreField("selected-model", fields["selected-model"]);
        renderModelDefaults(true);
        if (snapshot.defaultsModel === $("[data-selected-model]").value) {
            const fieldsRoot = $("[data-model-defaults-fields]");
            for (const [name, value] of Object.entries(snapshot.defaults || {})) {
                const input = fieldsRoot.querySelector('[data-default-field="' + name + '"]');
                if (!input) continue;
                if (Array.isArray(value)) {
                    const vectors = input.querySelectorAll("[data-vector-index]");
                    vectors.forEach((element, index) => {
                        if (value[index] != null) element.value = String(value[index]);
                    });
                } else if (input.tagName !== "SELECT" ||
                           [...input.options].some((option) => option.value === value)) {
                    input.value = value;
                }
            }
            updateDefaultFieldVisibility();
        }

        restoreField("test-model", fields["test-model"]);
        updateTestModelNote();
        // Speaker options vary by model and must exist before setting the voice.
        try { await refreshVoices(); } catch (error) {
            if (!destroyed) feedback(error.message, true);
        }
        if (destroyed) return;
        for (const [name, value] of Object.entries(fields)) {
            if (name === "selected-model" || name === "test-model" ||
                name === "test-voice" || name === "index-emo-reference") continue;
            restoreField(name, value);
        }
        restoreField("test-voice", fields["test-voice"]);
        restoreField("index-emo-reference", fields["index-emo-reference"]);
        for (const [emotion, value] of Object.entries(snapshot.emotions || {})) {
            const input = root.querySelector('[data-emotion="' + emotion + '"]');
            if (input) input.value = value;
        }
        updateEmotionControls();
        showVoices();
        root.querySelectorAll('[data-panel="voices"] > details').forEach((details, index) => {
            details.open = Boolean(snapshot.voiceAccordions?.[index]);
        });
        $("[data-index-emotion]").open = Boolean(snapshot.indexAccordion);

        const tab = ["server", "voices", "test"].includes(snapshot.tab)
            ? snapshot.tab : "server";
        root.querySelectorAll("[data-tab]").forEach((button) =>
            button.classList.toggle("is-active", button.dataset.tab === tab));
        root.querySelectorAll("[data-panel]").forEach((panel) =>
            panel.hidden = panel.dataset.panel !== tab);

        if (snapshot.lastAudioBlob instanceof Blob) {
            lastAudioBlob = snapshot.lastAudioBlob;
            if (audioURL) URL.revokeObjectURL(audioURL);
            audioURL = URL.createObjectURL(lastAudioBlob);
            $("[data-audio]").src = audioURL;
            $("[data-audio]").hidden = false;
        }
        window.requestAnimationFrame(() => {
            if (!destroyed) {
                container.scrollTop = Number(snapshot.containerScroll) || 0;
                root.scrollTop = Number(snapshot.rootScroll) || 0;
            }
        });
        updateModelDownloadButton();
    }
    let cloneRecorder = null;
    let cloneStream = null;
    let cloneRecordingFile = null;
    let cloneRecordingURL = null;
    let cloneRecordingSession = 0;
    let cloneRequestingMic = false;
    let cloneEncoding = false;
    let cloneSaving = false;
    let cloneStartTime = 0;
    let cloneDurationTimer = null;
    let cloneLimitTimer = null;

    function feedback(message, isError = false) {
        if (destroyed) return;
        const element = $(".tts-feedback");
        element.hidden = !message;
        element.textContent = message || "";
        element.classList.toggle("is-error", isError);
    }
    function setCloneRecordingStatus(message) {
        $("[data-clone-record-status]").textContent = message;
    }

    function updateCloneRecordingControls() {
        const recording = cloneRecorder?.state === "recording";
        const busy = cloneRequestingMic || cloneEncoding;
        const recordButton = $("[data-action=clone-record]");
        recordButton.textContent = recording ? "Stop recording"
            : cloneRequestingMic ? "Opening microphone…"
            : cloneEncoding ? "Preparing WAV…" : "Record voice";
        recordButton.disabled = (busy && !recording) || cloneSaving;
        $("[data-action=clone-discard]").hidden = !cloneRecordingFile;
        $("[data-action=clone-discard]").disabled = busy || cloneSaving;
        $("[data-action=clone]").disabled = recording || busy || cloneSaving;
        $("[data-clone-file]").disabled = recording || busy || cloneSaving;
        $("[data-clone-record-preview]").hidden = !cloneRecordingFile;
    }

    function clearCloneTimers() {
        if (cloneDurationTimer !== null) window.clearInterval(cloneDurationTimer);
        if (cloneLimitTimer !== null) window.clearTimeout(cloneLimitTimer);
        cloneDurationTimer = null;
        cloneLimitTimer = null;
    }

    function releaseCloneMicrophone() {
        if (cloneStream) {
            cloneStream.getTracks().forEach((track) => track.stop());
            cloneStream = null;
        }
    }

    function discardCloneRecording() {
        cloneRecordingFile = null;
        const player = $("[data-clone-record-preview]");
        player.pause();
        player.removeAttribute("src");
        player.load();
        if (cloneRecordingURL) URL.revokeObjectURL(cloneRecordingURL);
        cloneRecordingURL = null;
        updateCloneRecordingControls();
    }

    function cancelCloneCapture() {
        cloneRecordingSession++;
        clearCloneTimers();
        if (cloneRecorder) {
            cloneRecorder.ondataavailable = null;
            cloneRecorder.onstop = null;
            cloneRecorder.onerror = null;
            if (cloneRecorder.state !== "inactive") {
                try { cloneRecorder.stop(); } catch (_) { /* Already stopped. */ }
            }
            cloneRecorder = null;
        }
        releaseCloneMicrophone();
        cloneRequestingMic = false;
        cloneEncoding = false;
        if (!destroyed) updateCloneRecordingControls();
    }

    function stopCloneRecording() {
        if (!cloneRecorder || cloneRecorder.state !== "recording") return;
        clearCloneTimers();
        cloneEncoding = true;
        setCloneRecordingStatus("Processing microphone audio…");
        cloneRecorder.stop();
        updateCloneRecordingControls();
    }

    async function startCloneRecording() {
        if (cloneRequestingMic || cloneEncoding || cloneSaving || cloneRecorder) return;
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
            throw new Error("Microphone recording requires HTTPS or localhost. On a LAN IP, use HTTPS.");
        }
        if (!window.MediaRecorder) {
            throw new Error("Microphone recording is not supported by this browser.");
        }
        const session = ++cloneRecordingSession;
        cloneRequestingMic = true;
        setCloneRecordingStatus("Requesting microphone permission…");
        updateCloneRecordingControls();
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true},
                video: false,
            });
            if (destroyed || session !== cloneRecordingSession) {
                stream.getTracks().forEach((track) => track.stop());
                return;
            }
            const types = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"];
            const mimeType = types.find((type) => MediaRecorder.isTypeSupported(type));
            const recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
            const chunks = [];
            recorder.ondataavailable = (event) => {
                if (event.data?.size) chunks.push(event.data);
            };
            recorder.onerror = () => {
                if (session !== cloneRecordingSession || destroyed) return;
                cancelCloneCapture();
                setCloneRecordingStatus("Microphone recording failed. Try again.");
            };
            recorder.onstop = async () => {
                if (destroyed || session !== cloneRecordingSession) return;
                clearCloneTimers();
                releaseCloneMicrophone();
                cloneRecorder = null;
                try {
                    const wav = await recordedAudioToWav(new Blob(chunks, {
                        type: recorder.mimeType || "audio/webm",
                    }));
                    if (destroyed || session !== cloneRecordingSession) return;
                    if (wav.size > 20 * 1024 * 1024) {
                        throw new Error("Recording exceeds the 20 MB upload limit.");
                    }
                    discardCloneRecording();
                    cloneRecordingFile = wav;
                    cloneRecordingURL = URL.createObjectURL(wav);
                    const player = $("[data-clone-record-preview]");
                    player.src = cloneRecordingURL;
                    setCloneRecordingStatus("Recording ready. Preview it, then save the voice.");
                } catch (error) {
                    if (destroyed || session !== cloneRecordingSession) return;
                    setCloneRecordingStatus("Unable to prepare recording: " + error.message);
                } finally {
                    if (!destroyed && session === cloneRecordingSession) {
                        cloneEncoding = false;
                        updateCloneRecordingControls();
                    }
                }
            };
            recorder.start();
            cloneRecorder = recorder;
            cloneStream = stream;
            $("[data-clone-file]").value = "";
            discardCloneRecording();
            cloneStartTime = Date.now();
            setCloneRecordingStatus("Recording… 0 seconds");
            cloneDurationTimer = window.setInterval(() => {
                if (session !== cloneRecordingSession || destroyed) return;
                const seconds = Math.floor((Date.now() - cloneStartTime) / 1000);
                setCloneRecordingStatus("Recording… " + seconds + " seconds (119 max)");
            }, 1000);
            cloneLimitTimer = window.setTimeout(() => {
                if (session === cloneRecordingSession && !destroyed) stopCloneRecording();
            }, 119000);
        } catch (error) {
            if (stream) stream.getTracks().forEach((track) => track.stop());
            if (session === cloneRecordingSession) {
                cancelCloneCapture();
                setCloneRecordingStatus(error.name === "NotAllowedError"
                    ? "Microphone permission denied. Allow access in your browser and retry."
                    : error.name === "NotFoundError"
                    ? "No microphone found. Connect one and retry."
                    : error.message || "Could not start microphone recording.");
            }
        } finally {
            if (session === cloneRecordingSession && !destroyed) {
                cloneRequestingMic = false;
                updateCloneRecordingControls();
            }
        }
    }

    function activeTab(name) {
        if (name !== "voices" && (cloneRecorder || cloneRequestingMic || cloneEncoding)) {
            cancelCloneCapture();
            setCloneRecordingStatus("Recording canceled because the Voices tab was closed.");
        }
        root.querySelectorAll("[data-tab]").forEach((tab) =>
            tab.classList.toggle("is-active", tab.dataset.tab === name));
        root.querySelectorAll("[data-panel]").forEach((panel) =>
            panel.hidden = panel.dataset.panel !== name);
        feedback("");
        if (name === "voices" || name === "test") refreshVoices().catch((e) => feedback(e.message, true));
        if (name === "server") refreshModels().catch(() => {});
        captureSidebarSession();
    }
    root.querySelectorAll("[data-tab]").forEach((tab) =>
        tab.addEventListener("click", () => activeTab(tab.dataset.tab)));


    async function refreshModels() {
        const data = await (await request("/models")).json();
        if (destroyed) return;
        modelList = data.models || modelList;
        const activeModel = data.active_model || "voxcpm2";
        const selectors = [$("[data-selected-model]"), $("[data-test-model]")];
        for (const selector of selectors) {
            const selected = modelChoicesInitialized ? selector.value : activeModel;
            selector.replaceChildren();
            for (const model of modelList) {
                selector.add(new Option(model.label || model.id, model.id));
            }
            if ([...selector.options].some((item) => item.value === selected)) {
                selector.value = selected;
            }
        }
        if (!modelChoicesInitialized && activeModel.startsWith("qwen3-tts")) {
            $("[data-design-model]").value = "qwen3-tts-1.7b-voicedesign";
        }
        modelChoicesInitialized = true;
        updateTestModelNote();
        updateModelDownloadButton();
        renderModelDefaults();
        if (!initializing) refreshVoices().catch(() => {});
    }

    function fillVoiceOptions(select, spec, modelId, value) {
        select.replaceChildren();
        const isEmotion = spec.name === "emotion_reference_voice";
        const needsSaved = isEmotion || modelId === "indextts-2.5";
        select.add(new Option(needsSaved ? "Select saved audio" : "Default voice",
            needsSaved ? "" : "default"));
        if (!isEmotion && modelId.startsWith("qwen3-tts") &&
            modelId.endsWith("customvoice")) {
            for (const speaker of qwenSpeakers) {
                select.add(new Option("Qwen · " + speaker, speaker));
            }
        }
        for (const voice of savedVoices) {
            if (!voice.reference_file) continue;
            select.add(new Option(voice.name || voice.id, voice.id));
        }
        if (value && !(needsSaved && value === "default") &&
            ![...select.options].some((o) => o.value === value)) {
            // Retain manually entered values or a deleted recording so the
            // user can correct them rather than silently substituting a voice.
            select.add(new Option("Unavailable · " + value, value));
        }
        select.value = needsSaved && value === "default" ? "" : (value || (needsSaved ? "" : "default"));
    }

    function refreshDefaultVoiceOptions() {
        const modelId = $("[data-selected-model]").value;
        const profile = defaultsByModel[modelId];
        if (!profile) return;
        for (const spec of profile.fields.filter((f) => f.kind === "audio_voice")) {
            const select = $("[data-model-defaults-fields]")
                .querySelector('[data-default-field="' + spec.name + '"]');
            if (select) fillVoiceOptions(select, spec, modelId, select.value);
        }
    }

    function updateDefaultFieldVisibility() {
        const modelId = $("[data-selected-model]").value;
        const profile = defaultsByModel[modelId];
        if (!profile) return;
        const form = $("[data-model-defaults-fields]");
        for (const spec of profile.fields) {
            const row = form.querySelector('[data-default-row="' + spec.name + '"]');
            if (!row) continue;
            const conditions = spec.visible_when || {};
            row.hidden = Object.entries(conditions).some(([field, allowed]) => {
                const source = form.querySelector('[data-default-field="' + field + '"]');
                return source && !allowed.includes(source.value);
            });
        }
    }

    function renderModelDefaults(force = false) {
        const target = $("[data-model-defaults-fields]");
        const selected = $("[data-selected-model]").value;
        const profile = defaultsByModel[selected];
        const save = $("[data-action=save-model-defaults]");
        const reset = $("[data-action=reset-model-defaults]");
        save.disabled = reset.disabled = !profile;
        if (!force && selected === renderedDefaultsModel &&
            profile && target.querySelector("[data-default-row]")) {
            refreshDefaultVoiceOptions();
            updateDefaultFieldVisibility();
            return;
        }
        target.replaceChildren();
        renderedDefaultsModel = selected;
        if (!profile) {
            const info = document.createElement("p");
            info.className = "tts-muted";
            info.textContent = "Model settings unavailable while the TTS API is offline.";
            target.append(info);
            return;
        }
        for (const spec of profile.fields) {
            const group = spec.kind === "vector" || spec.kind === "audio_voice";
            const label = document.createElement(group ? "div" : "label");
            label.className = "tts-default-field";
            label.dataset.defaultRow = spec.name;
            if (group || spec.kind === "textarea") label.classList.add("is-wide");
            const name = document.createElement("span");
            name.textContent = spec.label;
            label.append(name);
            const value = profile.values[spec.name] ?? spec.default;
            let control;
            if (spec.kind === "vector") {
                control = document.createElement("div");
                control.className = "tts-emotion-grid";
                spec.emotions.forEach((emotion, index) => {
                    const emotionLabel = document.createElement("label");
                    emotionLabel.textContent = emotion[0].toUpperCase() + emotion.slice(1);
                    const input = document.createElement("input");
                    input.type = "number";
                    input.min = String(spec.min);
                    input.max = String(spec.max);
                    input.step = String(spec.step);
                    input.dataset.vectorIndex = String(index);
                    input.value = String(value[index]);
                    emotionLabel.append(input);
                    control.append(emotionLabel);
                });
            } else if (spec.kind === "audio_voice") {
                control = document.createElement("select");
                fillVoiceOptions(control, spec, selected, String(value || ""));
                label.append(control);
            } else if (spec.kind === "select") {
                control = document.createElement("select");
                spec.options.forEach((choice) => {
                    const names = {
                        emotion_mode: {neutral: "Neutral", auto: "Automatic emotion analysis",
                            text: "Emotion description", vector: "Manual emotion vector",
                            reference: "Reference audio"},
                        runtime_device: {auto: "Auto (CUDA preferred)", cpu: "CPU", cuda: "CUDA GPU"},
                        runtime_precision: {native: "Native (BF16 / FP32)",
                            int8: "INT8 (W8A8 · experimental)"},
                    };
                    const text = names[spec.name]?.[choice] || choice;
                    control.add(new Option(text, choice));
                });
                control.value = String(value);
                if (spec.name === "emotion_mode") {
                    control.addEventListener("change", updateDefaultFieldVisibility);
                }
            } else if (spec.kind === "textarea") {
                control = document.createElement("textarea");
                control.rows = 2;
                control.maxLength = spec.max_length || 1000;
                control.value = String(value);
            } else {
                control = document.createElement("input");
                control.type = ["integer", "nullable_integer", "number"].includes(spec.kind)
                    ? "number" : "text";
                if (spec.min !== undefined) control.min = String(spec.min);
                if (spec.max !== undefined) control.max = String(spec.max);
                if (spec.step !== undefined) control.step = String(spec.step);
                if (spec.max_length !== undefined) control.maxLength = spec.max_length;
                if (spec.kind === "integer" || spec.kind === "nullable_integer") control.step = "1";
                control.value = value == null ? "" : String(value);
            }
            control.dataset.defaultField = spec.name;
            if (spec.kind !== "audio_voice") label.append(control);
            if (spec.hint) {
                const hint = document.createElement("small");
                hint.className = "tts-muted";
                hint.textContent = spec.hint;
                label.append(hint);
            }
            target.append(label);
        }
        // Use visibility instead of rebuilding when changing emotion mode:
        // values for hidden modes remain intact while exploring the menu.
        updateDefaultFieldVisibility();
    }

    async function registerAudioFile(file, consent, label) {
        if (!consent) {
            throw new Error("Confirm permission to use " + label + " before uploading.");
        }
        if (!file || !file.size || file.size > 20 * 1024 * 1024) {
            throw new Error("Choose a valid audio recording no larger than 20 MB.");
        }
        const form = new FormData();
        const friendlyName = file.name.replace(/\.[^.]+$/, "").slice(0, 120) || "TTS reference";
        form.append("name", friendlyName);
        form.append("reference_text", "");
        form.append("consent", "true");
        form.append("audio", file);
        const voice = await (await request("/voices/clone", {
            method: "POST", body: form,
        })).json();
        savedVoices.push(voice);
        return voice;
    }

    function readModelDefaults() {
        const modelId = $("[data-selected-model]").value;
        const profile = defaultsByModel[modelId];
        if (!profile) throw new Error("Model settings are not available yet.");
        const values = {};
        for (const spec of profile.fields) {
            const input = $("[data-model-defaults-fields]")
                .querySelector('[data-default-field="' + spec.name + '"]');
            if (!input) throw new Error("Missing setting: " + spec.label);
            if (spec.kind === "vector") {
                values[spec.name] = [...input.querySelectorAll("[data-vector-index]")]
                    .map((field) => field.value.trim() === "" ? NaN : Number(field.value));
                if (values[spec.name].some((value) => !Number.isFinite(value))) {
                    throw new Error("Enter all eight emotion vector values.");
                }
            } else if (spec.kind === "nullable_integer" && input.value.trim() === "") {
                values[spec.name] = null;
            } else if (spec.kind === "integer" || spec.kind === "nullable_integer" ||
                       spec.kind === "number") {
                if (input.value.trim() === "") {
                    throw new Error("Enter " + spec.label + ".");
                }
                values[spec.name] = Number(input.value);
                if (!Number.isFinite(values[spec.name])) {
                    throw new Error("Invalid number for " + spec.label + ".");
                }
            } else {
                values[spec.name] = input.value;
            }
        }
        if (modelId === "indextts-2.5") {
            if (!values.voice) {
                throw new Error("IndexTTS requires a saved voice. Add one under Voices.");
            }
            if (values.emotion_mode === "text" && !values.instructions.trim()) {
                throw new Error("Enter an emotion description for Text mode.");
            }
            if (values.emotion_mode === "reference" &&
                !values.emotion_reference_voice) {
                throw new Error("Select a saved emotion reference voice.");
            }
        }
        return {model_id: modelId, settings: values};
    }

    function applyTestDefaults() {
        const profile = defaultsByModel[$("[data-test-model]").value];
        if (!profile) return;
        const values = profile.values;
        const put = (selector, key) => {
            if (key in values) {
                const control = $(selector);
                if (control) control.value = values[key] == null ? "" : String(values[key]);
            }
        };
        put("[data-test-language]", "language");
        put("[data-test-seed]", "seed");
        put("[data-test-steps]", "steps");
        put("[data-test-delivery]", "instructions");
        if (!("instructions" in values)) {
            $("[data-test-delivery]").value = "";
        }
        put("[data-index-emo-mode]", "emotion_mode");
        put("[data-index-emo-strength]", "emotion_strength");
        put("[data-index-duration]", "duration_factor");
        if (Array.isArray(values.emotion_vector)) {
            emotionNames.forEach((name, index) => {
                const control = root.querySelector('[data-emotion="' + name + '"]');
                if (control) control.value = String(values.emotion_vector[index]);
            });
        }
        const emotionReference = $("[data-index-emo-reference]");
        if (values.emotion_reference_voice &&
            [...emotionReference.options].some((o) => o.value === values.emotion_reference_voice)) {
            emotionReference.value = values.emotion_reference_voice;
        }
        const voice = $("[data-test-voice]");
        // Test starts with the saved Server-pane default for this model.
        // Keep the fallback when the configured recording has been removed.
        const configuredVoice = values.voice || "default";
        if ([...voice.options].some((o) => o.value === configuredVoice)) {
            voice.value = configuredVoice;
        }
        updateEmotionControls();
    }

    async function refreshModelDefaults() {
        const data = await (await request("/models/defaults")).json();
        if (destroyed) return;
        defaultsByModel = data.models || {};
        renderModelDefaults(true);
        applyTestDefaults();
    }

    function updateModelDownloadButton() {
        const id = $("[data-selected-model]").value;
        const button = $("[data-action=download-model]");
        const model = modelList.find((entry) => entry.id === id);
        const available = id === "voxcpm2" || model?.cache_detected === true;
        const downloading = downloadStatus?.state === "downloading";
        button.hidden = available;
        button.disabled = downloading;
        button.textContent = downloading && downloadStatus.model_id === id
            ? "Downloading weights…"
            : id === "indextts-2.5" ? "Download IndexTTS 2.5 weights"
            : "Download Qwen weights";
    }

    function updateEmotionControls() {
        const mode = $("[data-index-emo-mode]").value;
        $("[data-index-emo-vector-row]").hidden = mode !== "vector";
        $("[data-index-emo-ref-row]").hidden = mode !== "reference";
        $("[data-index-emo-file-row]").hidden = mode !== "reference";
        $("[data-index-emo-file-consent-row]").hidden =
            mode !== "reference" || !$("[data-index-emo-file]").files?.length;
    }

    function updateTestModelNote() {
        const id = $("[data-test-model]").value;
        const note = $("[data-test-model-note]");
        const qwen = id.startsWith("qwen3-tts");
        const index = id === "indextts-2.5";
        $("[data-index-emotion]").hidden = !index;
        $("[data-test-steps]").disabled = qwen || index;
        $("[data-test-steps]").title = qwen || index
            ? "This backend does not use VoxCPM inference steps"
            : "VoxCPM2 inference steps";
        if (index) {
            note.textContent = "IndexTTS requires a saved voice recording. English, Chinese, Japanese, Spanish, Arabic (Auto chooses English). Emotion controls are optional.";
        } else if (id.endsWith("-base")) {
            note.textContent = "Base clones saved recordings only. Delivery instructions and VoxCPM steps are not available.";
        } else if (id.endsWith("-voicedesign")) {
            note.textContent = "VoiceDesign uses delivery instructions as a separate voice description. Saved voices use Base automatically.";
        } else if (id.startsWith("qwen3-tts")) {
            note.textContent = id.includes("0.6b")
                ? "0.6B CustomVoice: nine built-in speakers. Delivery instructions are not supported; saved voices use Base."
                : "1.7B CustomVoice: nine built-in speakers with delivery control. Saved voice recordings use Base and cannot take delivery instructions.";
        } else {
            note.textContent = "VoxCPM2: use a saved character voice or default. Voice/delivery style can use a short cached reference.";
        }
    }

    async function refreshDownloadStatus() {
        const status = await (await request("/models/download/status")).json();
        if (destroyed) return;
        downloadStatus = status;
        updateModelDownloadButton();
        if (status.state === "complete" && latestCompletedDownload !== status.model_id) {
            latestCompletedDownload = status.model_id;
            refreshModels().catch(() => {});
        }
        const box = $("[data-download-status]");
        if (status.state === "downloading") box.textContent = "Downloading " + status.model_id + " under models/TTS. Check TTS worker logs for progress.";
        else if (status.state === "complete") box.textContent = "Downloaded: " + status.model_id;
        else if (status.state === "failed") box.textContent = "Download failed for " + status.model_id + ": " + status.error;
        else box.textContent = "Weights stored in ComfyUI/models/TTS. IndexTTS needs Download weights first.";
    }

    async function refreshStatus() {
        const result = await (await request("/status")).json();
        if (destroyed) return;
        const serverChanged = rememberComfyuiSession(result.comfyui_session);
        if (result.comfyui_session) sessionVerified = true;
        if (serverChanged && !initializing) {
            // ComfyUI restarted while the browser stayed open. Do not let
            // destroy() overwrite the cleared snapshot with stale fields.
            skipSessionCapture = true;
            mount(container);
            return;
        }
        const h = result.health;
        const ok = Boolean(h && h.ok);
        $(".tts-status-dot").className = "tts-status-dot " + (ok ? "is-online" : "is-offline");
        $(".tts-banner").textContent = !result.installed
            ? "TTS backend not installed. Run bash install.sh once."
            : !ok ? "API is offline. Check the ComfyUI console or managed-server.log."
            : h.paused ? "Paused — ComfyUI is using the GPU."
            : h.loaded ? "Online · Speech model loaded" : "Online · Model loads on demand";
        $("[data-service]").textContent = ok ? (result.running ? "Managed · online" : "External · online") : "Offline";
        $("[data-model]").textContent = h?.loaded_model || h?.active_model || h?.model || "VoxCPM2";
        const modelSelected = $("[data-selected-model]").value;
        $("[data-model-hint]").textContent = h?.index_installed === false && modelSelected === "indextts-2.5"
            ? "IndexTTS missing: run bash install-index.sh (isolated Python 3.11), then Download weights."
            : h?.qwen_installed === false && modelSelected.startsWith("qwen3-tts")
            ? "Qwen runtime missing: run bash install-qwen.sh in the TTS custom node folder, then restart its API. Model weights download separately."
            : "Selecting a default restarts only the TTS service. Models are stored under ComfyUI/models/TTS.";
        $("[data-action=apply-model]").disabled = result.active_model_editable === false;
        $("[data-version]").textContent = h?.version || "Older release";
        const runtime = h?.runtime;
        const device = runtime?.device || "unknown";
        $("[data-memory]").textContent = !ok ? "Unavailable" : h.paused
            ? "Released / paused" : !h.loaded ? "Unloaded"
            : device === "cpu" ? "Loaded in RAM" : device === "auto" ? "Loaded" : "Loaded in VRAM";
        $("[data-runtime]").textContent = !ok ? "Unavailable" : !runtime
            ? "Not loaded"
            : (device === "cuda:0" ? "CUDA" : device.toUpperCase()) +
              " · " + (runtime.precision === "int8" ? "INT8 W8A8" : "Native") +
              (runtime.precision === "int8" ? " · " + runtime.int8_layers + " layers" : "");
        $("[data-allocator]").textContent = h?.cuda_allocator || "Unknown";
        $("[data-idle]").textContent = result.config.idle_unload_seconds === 0
            ? "Disabled" : result.config.idle_unload_seconds + "s";
        $("[data-queue]").textContent = result.busy ? "Busy" : "Idle";
        $("[data-endpoint]").textContent = result.api_url + "/v1/audio/speech";
        const key = result.api_key || {};
        $("[data-api-key-status]").textContent = key.configured ? "Enabled" : "Disabled (localhost)";
        root.querySelectorAll("[data-action=generate-key], [data-action=save-key]").forEach((button) => {
            button.disabled = key.editable === false;
            button.title = key.editable === false ? "API key managed by environment variable" : "";
        });
        const bench = h?.last_stats;
        $("[data-bench]").textContent = bench?.seconds != null
            ? `${bench.seconds}s generated ${bench.audio_seconds}s audio · RTF ${bench.real_time_factor ?? "?"}`
            : "No speech generated yet.";
    }

    function cleanupVoicePreviews() {
        for (const url of voicePreviewUrls.values()) URL.revokeObjectURL(url);
        voicePreviewUrls.clear();
    }

    function voiceUsage(voiceId) {
        const usages = [];
        for (const [modelId, profile] of Object.entries(defaultsByModel)) {
            for (const field of ["voice", "emotion_reference_voice"]) {
                if (profile?.values?.[field] === voiceId) {
                    usages.push(modelId + (field === "voice" ? " (default)" : " (emotion)"));
                }
            }
        }
        return usages;
    }

    function showVoices() {
        const list = $("[data-voices]");
        list.replaceChildren();
        cleanupVoicePreviews();
        const search = $("[data-voice-search]").value.trim().toLocaleLowerCase();
        const sorted = [...savedVoices].sort((left, right) =>
            (left.name || "").localeCompare(right.name || "", undefined,
                {sensitivity: "base", numeric: true})
        );
        const matches = sorted.filter((voice) =>
            [voice.name, voice.id, voice.description, voice.reference_text]
                .some((value) => String(value || "").toLocaleLowerCase().includes(search)));
        $("[data-voice-count]").textContent = search
            ? matches.length + " of " + sorted.length + " saved voices"
            : sorted.length + " saved " + (sorted.length === 1 ? "voice" : "voices");
        if (!matches.length) {
            const empty = document.createElement("p");
            empty.className = "tts-muted";
            empty.textContent = sorted.length
                ? "No voices match your search."
                : "No saved voices. Clone a recording or design a voice below.";
            list.append(empty);
        }

        for (const voice of matches) {
            const card = document.createElement("article");
            card.className = "tts-voice";
            const details = document.createElement("div");
            details.className = "tts-voice-details";
            const title = document.createElement("strong");
            title.textContent = voice.name || voice.id;
            const code = document.createElement("code");
            code.textContent = voice.id;
            details.append(title, code);
            if (voice.description) {
                const description = document.createElement("span");
                description.className = "tts-voice-description";
                description.textContent = voice.description;
                details.append(description);
            }
            const usage = voiceUsage(voice.id);
            if (usage.length) {
                const inUse = document.createElement("small");
                inUse.className = "tts-muted";
                inUse.textContent = "Used by " + usage.join(", ");
                details.append(inUse);
            }
            const actions = document.createElement("div");
            actions.className = "tts-voice-actions";
            const editor = document.createElement("div");
            editor.className = "tts-voice-editor";
            editor.hidden = true;

            const editButton = document.createElement("button");
            editButton.type = "button";
            editButton.textContent = "View / Edit";
            editButton.setAttribute("aria-expanded", "false");
            const editorId = "tts-voice-" + voice.id;
            editor.id = editorId;
            editButton.setAttribute("aria-controls", editorId);
            editButton.addEventListener("click", () => {
                editor.hidden = !editor.hidden;
                if (editor.hidden) editor.querySelector("audio")?.pause();
                editButton.setAttribute("aria-expanded", String(!editor.hidden));
                editButton.textContent = editor.hidden ? "View / Edit" : "Close details";
            });

            const testButton = document.createElement("button");
            testButton.type = "button";
            testButton.textContent = "Use in Test";
            testButton.addEventListener("click", () => {
                if (!voice.reference_file) {
                    feedback("This voice has no saved recording to use as a speech reference.", true);
                    return;
                }
                $("[data-test-voice]").value = voice.id;
                activeTab("test");
            });

            const copyButton = document.createElement("button");
            copyButton.type = "button";
            copyButton.textContent = "Copy ID";
            copyButton.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(voice.id);
                    feedback("Voice ID copied.");
                } catch (_) {
                    feedback("Could not copy voice ID.", true);
                }
            });

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.textContent = "Delete";
            removeButton.className = "tts-destructive";
            removeButton.addEventListener("click", async () => {
                const warning = usage.length
                    ? "\n\nThis voice is referenced in model defaults: " +
                      usage.join(", ") + ". You may need to select another voice."
                    : "";
                if (!window.confirm("Delete saved voice \"" + voice.name +
                    "\" and its reference recording? This cannot be undone." + warning)) return;
                removeButton.disabled = true;
                try {
                    await request("/voices/" + encodeURIComponent(voice.id),
                                  {method: "DELETE"});
                    await refreshVoices();
                    feedback("Voice deleted.");
                } catch (error) {
                    removeButton.disabled = false;
                    feedback(error.message, true);
                }
            });
            actions.append(editButton, testButton, copyButton, removeButton);

            const addField = (text, value, type = "input", limit = 1000) => {
                const label = document.createElement("label");
                label.textContent = text;
                const input = document.createElement(type === "textarea" ? "textarea" : "input");
                if (type === "textarea") input.rows = 2;
                else input.type = "text";
                input.maxLength = limit;
                input.value = value || "";
                label.append(input);
                editor.append(label);
                return input;
            };
            const name = addField("Voice name", voice.name, "input", 120);
            const description = addField("Description", voice.description, "textarea", 1000);
            const referenceText = addField("Reference transcript", voice.reference_text, "textarea", 4000);

            if (voice.reference_file) {
                const listenButton = document.createElement("button");
                listenButton.type = "button";
                listenButton.textContent = "Preview reference recording";
                const player = document.createElement("audio");
                player.controls = true;
                player.preload = "none";
                player.hidden = true;
                player.addEventListener("play", () => {
                    root.querySelectorAll(".tts-voice-editor audio").forEach((other) => {
                        if (other !== player) other.pause();
                    });
                });
                listenButton.addEventListener("click", async () => {
                    if (!player.hidden) {
                        player.pause();
                        player.hidden = true;
                        listenButton.textContent = "Preview reference recording";
                        return;
                    }
                    if (!player.src) {
                        listenButton.disabled = true;
                        listenButton.textContent = "Loading recording…";
                        try {
                            const response = await request("/voices/" +
                                encodeURIComponent(voice.id) + "/audio");
                            const blob = await response.blob();
                            if (destroyed || !card.isConnected) return;
                            const url = URL.createObjectURL(blob);
                            voicePreviewUrls.set(voice.id, url);
                            player.src = url;
                        } catch (error) {
                            feedback(error.message, true);
                            return;
                        } finally {
                            listenButton.disabled = false;
                            if (!player.src) listenButton.textContent = "Preview reference recording";
                        }
                    }
                    player.hidden = false;
                    listenButton.textContent = "Hide reference recording";
                });
                editor.append(listenButton, player);
            } else {
                const note = document.createElement("p");
                note.className = "tts-muted";
                note.textContent = "No recording is attached. Add one below to enable cloning.";
                editor.append(note);
            }

            const fileLabel = document.createElement("label");
            fileLabel.className = "tts-audio-picker";
            fileLabel.textContent = voice.reference_file
                ? "Replace reference recording (optional)"
                : "Add reference recording (optional)";
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = "audio/*,.wav,.flac,.ogg,.mp3,.aiff";
            fileLabel.append(fileInput);
            editor.append(fileLabel);
            const permissionLabel = document.createElement("label");
            permissionLabel.className = "tts-check";
            permissionLabel.hidden = true;
            const consent = document.createElement("input");
            consent.type = "checkbox";
            permissionLabel.append(consent, document.createTextNode(
                " I have permission to use the replacement recording."));
            editor.append(permissionLabel);
            fileInput.addEventListener("change", () => {
                permissionLabel.hidden = !fileInput.files?.length;
                consent.checked = false;
            });
            const saveButton = document.createElement("button");
            saveButton.type = "button";
            saveButton.className = "tts-primary";
            saveButton.textContent = "Save voice changes";
            saveButton.addEventListener("click", async () => {
                const newName = name.value.trim();
                if (!newName) return feedback("Enter a voice name.", true);
                const file = fileInput.files?.[0];
                if (file && !consent.checked) {
                    return feedback("Confirm permission to use the new recording.", true);
                }
                if (file && file.size > 20 * 1024 * 1024) {
                    return feedback("Recording exceeds the 20 MB limit.", true);
                }
                const body = new FormData();
                body.append("name", newName);
                body.append("description", description.value);
                body.append("reference_text", referenceText.value);
                body.append("consent", String(Boolean(file && consent.checked)));
                if (file) body.append("audio", file);
                saveButton.disabled = true;
                saveButton.textContent = "Saving…";
                try {
                    const updated = await (await request(
                        "/voices/" + encodeURIComponent(voice.id),
                        {method: "PATCH", body}
                    )).json();
                    await refreshVoices();
                    feedback("Voice saved. ID unchanged: " + updated.id);
                } catch (error) {
                    saveButton.disabled = false;
                    saveButton.textContent = "Save voice changes";
                    feedback(error.message, true);
                }
            });
            editor.append(saveButton);
            card.append(details, actions, editor);
            list.append(card);
        }
    }

    async function refreshVoices() {
        const voices = await (await request("/voices")).json();
        if (destroyed) return;
        savedVoices = voices;
        refreshDefaultVoiceOptions();
        const select = $("[data-test-voice]");
        const previous = select.value;
        const emotionSelect = $("[data-index-emo-reference]");
        const previousEmotion = emotionSelect.value;
        emotionSelect.replaceChildren(new Option("Select saved voice", "default"));
        select.replaceChildren(new Option("Default voice", "default"));
        const model = $("[data-test-model]").value;
        if (model.startsWith("qwen3-tts") && model.endsWith("customvoice")) {
            for (const speaker of qwenSpeakers) {
                select.add(new Option("Qwen · " + speaker, speaker));
            }
        }
        for (const voice of [...voices].sort((a, b) =>
            (a.name || "").localeCompare(b.name || "", undefined, {sensitivity: "base"}))) {
            select.add(new Option(voice.name || voice.id, voice.id));
            if (voice.reference_file) {
                emotionSelect.add(new Option(voice.name || voice.id, voice.id));
            }
        }
        if ([...select.options].some((option) => option.value === previous)) {
            select.value = previous;
        }
        if ([...emotionSelect.options].some((option) => option.value === previousEmotion)) {
            emotionSelect.value = previousEmotion;
        }
        if ((model.endsWith("-base") || model === "indextts-2.5") &&
            select.value === "default" && voices.some((v) => v.reference_file)) {
            select.value = voices.find((v) => v.reference_file).id;
        }
        showVoices();
    }

    async function perform(button, operation, success) {
        button.disabled = true;
        feedback("");
        try {
            await operation();
            if (success) feedback(success);
        } catch (e) {
            feedback(e.message || "Operation failed.", true);
        } finally {
            if (!destroyed) {
                button.disabled = false;
                if (button.matches('[data-action="download-model"]')) {
                    updateModelDownloadButton();
                }
            }
            refreshStatus().catch(() => {});
        }
    }

    $("[data-action=refresh]").addEventListener("click", () =>
        refreshStatus().catch((e) => feedback(e.message, true)));
    $("[data-action=refresh-voices]").addEventListener("click", () =>
        refreshVoices().catch((e) => feedback(e.message, true)));
    $("[data-voice-search]").addEventListener("input", showVoices);
    $("[data-action=copy-url]").addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText($("[data-endpoint]").textContent);
            feedback("Speech API URL copied.");
        } catch (e) { feedback("Copy failed.", true); }
    });
    $("[data-action=unload]").addEventListener("click", (e) => perform(
        e.currentTarget, () => request("/model/unload", { method: "POST" }),
        "Speech model unloaded."));
    $("[data-action=restart]").addEventListener("click", (e) => perform(
        e.currentTarget, () => request("/server/restart", { method: "POST" }),
        "Speech API is restarting."));
    $("[data-test-model]").addEventListener("change", () => {
        updateTestModelNote();
        refreshVoices().then(applyTestDefaults)
            .catch((err) => feedback(err.message, true));
    });
    $("[data-selected-model]").addEventListener("change", () => {
        updateModelDownloadButton();
        renderModelDefaults();
        // Refresh on-disk download state, including auto-downloads triggered
        // outside the sidebar, whenever the selected model changes.
        refreshModels().catch(() => {});
    });
    $("[data-action=save-model-defaults]").addEventListener("click", (e) => perform(
        e.currentTarget, async () => {
            const payload = readModelDefaults();
            const response = await (await request("/models/defaults", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            })).json();
            defaultsByModel[payload.model_id].values = response.settings;
            renderModelDefaults(true);
            await refreshVoices();
            if ($("[data-test-model]").value === payload.model_id) applyTestDefaults();
        }, "Defaults saved for this model. New API requests use them unless explicitly overridden."
    ));
    $("[data-action=reset-model-defaults]").addEventListener("click", (e) => perform(
        e.currentTarget, async () => {
            const modelId = $("[data-selected-model]").value;
            const response = await (await request("/models/defaults", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({model_id: modelId, settings: {}, reset: true}),
            })).json();
            defaultsByModel[modelId].values = response.settings;
            renderModelDefaults(true);
            if ($("[data-test-model]").value === modelId) applyTestDefaults();
        }, "Built-in defaults restored for this model."
    ));
    $("[data-index-emo-mode]").addEventListener("change", updateEmotionControls);
    $("[data-index-emo-file]").addEventListener("change", () => {
        $("[data-index-emo-file-consent]").checked = false;
        updateEmotionControls();
    });
    updateEmotionControls();
    $("[data-action=apply-model]").addEventListener("click", (e) => perform(
        e.currentTarget, async () => {
            const id = $("[data-selected-model]").value;
            const result = await (await request("/settings/model", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id: id }),
            })).json();
            $("[data-test-model]").value = id;
            updateTestModelNote();
            refreshVoices().catch(() => {});
            feedback("Default model set to " + result.active_model + ". The TTS API is restarting; ComfyUI stays running.");
        }
    ));
    $("[data-action=download-model]").addEventListener("click", (e) => perform(
        e.currentTarget, async () => {
            const id = $("[data-selected-model]").value;
            if (!id.startsWith("qwen3-tts") && id !== "indextts-2.5") {
                throw new Error("Choose a Qwen3-TTS or IndexTTS model to download.");
            }
            latestCompletedDownload = null;
            await request("/models/download", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id: id }),
            });
            await refreshDownloadStatus();
        }
    ));


    // API keys are stored server-side in .env; only a newly generated key is
    // returned once. Never include credentials in passive status responses.
    $("[data-action=generate-key]").addEventListener("click", (e) => perform(
        e.currentTarget, async () => {
            const result = await (await request("/settings/api-key", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({action: "generate"}),
            })).json();
            $("[data-api-key]").value = result.key;
            feedback("Key generated and saved. Copy it now for SillyTavern; it will not be shown again after closing the panel.");
        }));
    $("[data-action=save-key]").addEventListener("click", (e) => perform(
        e.currentTarget, async () => {
            const key = $("[data-api-key]").value.trim();
            if (key.length < 24) throw new Error("API key must contain at least 24 characters.");
            await request("/settings/api-key", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({action: "set", key}),
            });
            feedback("API key saved. The managed service is restarting. Update SillyTavern with this key.");
        }));
    $("[data-action=copy-key]").addEventListener("click", async () => {
        const key = $("[data-api-key]").value;
        if (!key) return feedback("Generate or enter an API key before copying.", true);
        try {
            await navigator.clipboard.writeText(key);
            feedback("API key copied.");
        } catch (_) {
            feedback("Copy failed. Select and copy the field manually.", true);
        }
    });
    $("[data-action=toggle-key]").addEventListener("click", () => {
        const input = $("[data-api-key]");
        input.type = input.type === "password" ? "text" : "password";
        $("[data-action=toggle-key]").textContent = input.type === "password" ? "Show" : "Hide";
    });

    $("[data-action=design]").addEventListener("click", (e) => perform(
        e.currentTarget,
        async () => {
            const body = {
                model: $("[data-design-model]").value,
                name: $("[data-design-name]").value.trim(),
                description: $("[data-design-description]").value.trim(),
                sample_text: $("[data-design-sample]").value,
                seed: Number($("[data-design-seed]").value),
            };
            if (!body.name || !body.description || !body.sample_text.trim()) {
                throw new Error("Enter a name, description, and sample text.");
            }
            const voice = await (await request("/voices/design", {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
            })).json();
            await refreshVoices();
            $("[data-test-voice]").value = voice.id;
            activeTab("test");
        }, "Voice created."));

    $("[data-action=clone-record]").addEventListener("click", () => {
        if (cloneRecorder?.state === "recording") {
            stopCloneRecording();
        } else {
            startCloneRecording().catch((error) =>
                setCloneRecordingStatus(error.message || "Could not access microphone."));
        }
    });
    $("[data-action=clone-discard]").addEventListener("click", () => {
        discardCloneRecording();
        setCloneRecordingStatus("Recording discarded.");
    });
    $("[data-clone-file]").addEventListener("change", () => {
        if ($("[data-clone-file]").files.length) {
            discardCloneRecording();
            setCloneRecordingStatus("");
        }
    });
    const onCloneVisibilityChange = () => {
        if (document.visibilityState === "hidden" &&
            (cloneRecorder || cloneRequestingMic || cloneEncoding)) {
            cancelCloneCapture();
            setCloneRecordingStatus("Recording canceled when the browser tab was hidden.");
        }
    };
    document.addEventListener("visibilitychange", onCloneVisibilityChange);
    updateCloneRecordingControls();

    $("[data-action=clone]").addEventListener("click", (e) => perform(
        e.currentTarget,
        async () => {
            if (cloneRecorder || cloneRequestingMic || cloneEncoding) {
                throw new Error("Stop recording and let the audio finish processing first.");
            }
            const file = cloneRecordingFile || $("[data-clone-file]").files[0];
            if (!file) throw new Error("Choose a reference audio file or record your voice.");
            const name = $("[data-clone-name]").value.trim();
            if (!name) throw new Error("Enter a character name.");
            if (!$("[data-clone-consent]").checked) throw new Error("Confirm authorization to use this voice.");
            cloneSaving = true;
            updateCloneRecordingControls();
            try {
                const form = new FormData();
                form.append("name", name);
                form.append("reference_text", $("[data-clone-transcript]").value);
                form.append("consent", "true");
                form.append("audio", file, file.name);
                const voice = await (await request("/voices/clone", {method: "POST", body: form})).json();
                discardCloneRecording();
                $("[data-clone-file]").value = "";
                $("[data-clone-consent]").checked = false;
                setCloneRecordingStatus("");
                await refreshVoices();
                $("[data-test-voice]").value = voice.id;
                activeTab("test");
            } finally {
                cloneSaving = false;
                updateCloneRecordingControls();
            }
        }, "Voice saved."));

    $("[data-action=generate]").addEventListener("click", (e) => perform(
        e.currentTarget,
        async () => {
            const body = {
                model: $("[data-test-model]").value,
                language: $("[data-test-language]").value,
                voice: $("[data-test-voice]").value,
                input: $("[data-test-text]").value,
                instructions: $("[data-test-delivery]").value.trim(),
                seed: $("[data-test-seed]").value.trim() === ""
                    ? null : Number($("[data-test-seed]").value),
                steps: Number($("[data-test-steps]").value),
            };
            if (body.model === "indextts-2.5") {
                body.emotion_mode = $("[data-index-emo-mode]").value;
                body.emotion_strength = Number($("[data-index-emo-strength]").value);
                body.duration_factor = Number($("[data-index-duration]").value);
                if (body.emotion_mode === "vector") {
                    body.emotion_vector = emotionNames.map(
                        (name) => Number(root.querySelector('[data-emotion="' + name + '"]').value));
                } else if (body.emotion_mode === "reference") {
                    body.emotion_reference_voice = $("[data-index-emo-reference]").value;
                }
                if (body.voice === "default") {
                    throw new Error("IndexTTS needs a saved voice. Upload a recording under Voices → Clone.");
                }
            }
            if (!body.input.trim()) throw new Error("Enter text to speak.");
            const emotionFile = $("[data-index-emo-file]").files?.[0];
            if (body.model === "indextts-2.5" && body.emotion_mode === "reference" &&
                emotionFile) {
                const voice = await registerAudioFile(
                    emotionFile, $("[data-index-emo-file-consent]").checked,
                    "emotion reference"
                );
                body.emotion_reference_voice = voice.id;
                $("[data-index-emo-file]").value = "";
                $("[data-index-emo-file-consent]").checked = false;
                await refreshVoices();
                $("[data-index-emo-reference]").value = voice.id;
                updateEmotionControls();
            }
            const response = await request("/speech", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const blob = await response.blob();
            lastAudioBlob = blob;
            if (audioURL) URL.revokeObjectURL(audioURL);
            cleanupVoicePreviews();
            audioURL = URL.createObjectURL(blob);
            const player = $("[data-audio]");
            player.src = audioURL;
            player.hidden = false;
            player.play().catch(() => {});
            captureSidebarSession();
        }, "Generation complete."));

    async function initializeSidebar() {
        // Verify the ComfyUI process before restoring even a single prior
        // field. A TTS worker restart does not change this identity.
        try { await refreshStatus(); }
        catch (error) { if (!destroyed) feedback(error.message, true); }
        if (destroyed) return;
        try { await refreshModels(); }
        catch (error) { if (!destroyed) feedback(error.message, true); }
        try { await refreshModelDefaults(); }
        catch (error) { if (!destroyed) feedback(error.message, true); }
        try { await refreshVoices(); }
        catch (error) { if (!destroyed) feedback(error.message, true); }
        if (destroyed) return;

        const snapshot = savedSidebarSession || storedSidebarSession();
        if (sessionVerified && snapshot &&
            snapshot.comfyuiSession === comfyuiSessionId) {
            await restoreSidebarSession(snapshot);
        } else if (sessionVerified && snapshot) {
            clearSidebarSession();
        }
        if (destroyed) return;
        initializing = false;
        captureSidebarSession();
    }

    initializeSidebar().catch((error) => {
        if (!destroyed) {
            initializing = false;
            feedback(error.message, true);
        }
    });
    refreshDownloadStatus().catch(() => {});
    timer = window.setInterval(() => {
        if (!destroyed && document.visibilityState !== "hidden") {
            refreshStatus().catch(() => {});
            refreshDownloadStatus().catch(() => {});
        }
    }, 4000);

    activePanel = {
        destroy() {
            if (!skipSessionCapture) captureSidebarSession();
            destroyed = true;
            cancelCloneCapture();
            discardCloneRecording();
            document.removeEventListener("visibilitychange", onCloneVisibilityChange);
            window.removeEventListener("pagehide", onPageHide);
            if (timer) window.clearInterval(timer);
            if (audioURL) URL.revokeObjectURL(audioURL);
            cleanupVoicePreviews();
        },
    };
}

app.registerExtension({
    name: "LatentTalent.TTS.API.Sidebar",
    async setup() {
        // ComfyUI serves extensions as native ES modules; CSS imports are not
        // supported by the browser without a bundler. Load the stylesheet.
        if (!document.getElementById("tts-api-sidebar-styles")) {
            const style = document.createElement("link");
            style.id = "tts-api-sidebar-styles";
            style.rel = "stylesheet";
            style.href = new URL("./tts_sidebar.css", import.meta.url).href;
            document.head.appendChild(style);
        }
        app.extensionManager.registerSidebarTab({
            id: "latent-talent-tts-api",
            title: "TTS API",
            tooltip: "Local speech service · VoxCPM2",
            icon: "pi pi-volume-up",
            type: "custom",
            render: mount,
            destroy() {
                if (activePanel) activePanel.destroy();
                activePanel = null;
            },
        });
    },
});
