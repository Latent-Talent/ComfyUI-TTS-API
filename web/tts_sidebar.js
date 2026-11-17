import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const ROUTE = "/tts-api";
let activePanel = null;

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
        <div class="tts-section-title">Inference service</div>
        <div class="tts-info">
          <div><span>Backend</span><strong data-model>VoxCPM2</strong></div>
          <div><span>API version</span><strong data-version>Unknown</strong></div>
          <div><span>API</span><strong data-service>Checking</strong></div>
          <div><span>Model memory</span><strong data-memory>Not loaded</strong></div>
          <div><span>CUDA allocator</span><strong data-allocator>Native</strong></div>
          <div><span>Idle unload</span><strong data-idle>60s</strong></div>
          <div><span>ComfyUI queue</span><strong data-queue>Checking</strong></div>
        </div>
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
      </div>
      <div class="tts-panel" data-panel="voices" hidden>
        <div class="tts-section-title">Saved character voices</div>
        <div class="tts-actions"><button type="button" data-action="refresh-voices">Refresh voices</button></div>
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

function mount(container) {
    if (activePanel) activePanel.destroy();
    container.innerHTML = template();
    const root = container.querySelector(".tts-ui");
    let timer = null;
    let audioURL = null;
    let destroyed = false;
    let modelList = [{ id: "voxcpm2", label: "VoxCPM2", kind: "voxcpm" }];
    let modelChoicesInitialized = false;
    const emotionNames = ["happy", "angry", "sad", "afraid", "disgusted",
                          "melancholic", "surprised", "calm"];
    const qwenSpeakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
                          "Ryan", "Aiden", "Ono_Anna", "Sohee"];
    const $ = (selector) => root.querySelector(selector);

    function feedback(message, isError = false) {
        if (destroyed) return;
        const element = $(".tts-feedback");
        element.hidden = !message;
        element.textContent = message || "";
        element.classList.toggle("is-error", isError);
    }
    function activeTab(name) {
        root.querySelectorAll("[data-tab]").forEach((tab) =>
            tab.classList.toggle("is-active", tab.dataset.tab === name));
        root.querySelectorAll("[data-panel]").forEach((panel) =>
            panel.hidden = panel.dataset.panel !== name);
        feedback("");
        if (name === "voices" || name === "test") refreshVoices().catch((e) => feedback(e.message, true));
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
        refreshVoices().catch(() => {});
    }

    function updateModelDownloadButton() {
        const id = $("[data-selected-model]").value;
        const button = $("[data-action=download-model]");
        button.disabled = id === "voxcpm2";
        button.textContent = id === "indextts-2.5"
            ? "Download IndexTTS 2.5 weights" : id.startsWith("qwen3-tts")
                ? "Download Qwen weights" : "Weights managed by VoxCPM";
    }

    function updateEmotionControls() {
        const mode = $("[data-index-emo-mode]").value;
        $("[data-index-emo-vector-row]").hidden = mode !== "vector";
        $("[data-index-emo-ref-row]").hidden = mode !== "reference";
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
        const box = $("[data-download-status]");
        if (status.state === "downloading") box.textContent = "Downloading " + status.model_id + " under models/TTS. Check TTS worker logs for progress.";
        else if (status.state === "complete") box.textContent = "Downloaded: " + status.model_id;
        else if (status.state === "failed") box.textContent = "Download failed for " + status.model_id + ": " + status.error;
        else box.textContent = "Weights stored in ComfyUI/models/TTS. IndexTTS needs Download weights first.";
    }

    async function refreshStatus() {
        const result = await (await request("/status")).json();
        if (destroyed) return;
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
        $("[data-memory]").textContent = !ok ? "Unavailable" : h.paused ? "Released / paused" : h.loaded ? "Loaded in VRAM" : "Unloaded";
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

    async function refreshVoices() {
        const voices = await (await request("/voices")).json();
        if (destroyed) return;
        const list = $("[data-voices]");
        list.replaceChildren();
        const select = $("[data-test-voice]");
        const choice = select.value;
        const emotionSelect = $("[data-index-emo-reference]");
        const previousEmotionRef = emotionSelect.value;
        emotionSelect.replaceChildren(new Option("Select saved voice", "default"));
        select.replaceChildren(new Option("Default voice", "default"));
        const chosenModel = $("[data-test-model]").value;
        if (chosenModel.startsWith("qwen3-tts") && chosenModel.endsWith("customvoice")) {
            for (const speaker of qwenSpeakers) {
                select.add(new Option("Qwen · " + speaker, speaker));
            }
        }
        if (!voices.length) {
            const empty = document.createElement("p");
            empty.className = "tts-muted";
            empty.textContent = "No saved voices. Design or clone one below.";
            list.append(empty);
        }
        for (const voice of voices) {
            const item = document.createElement("div");
            item.className = "tts-voice";
            const details = document.createElement("div");
            details.className = "tts-voice-details";
            const title = document.createElement("strong");
            title.textContent = voice.name || voice.id;
            const code = document.createElement("code");
            code.textContent = voice.id;
            code.title = voice.description || voice.id;
            details.append(title, code);
            const tools = document.createElement("div");
            tools.className = "tts-voice-actions";
            const copy = document.createElement("button");
            copy.type = "button";
            copy.textContent = "Copy ID";
            copy.addEventListener("click", () => navigator.clipboard.writeText(voice.id)
                .then(() => feedback("Copied voice ID."))
                .catch(() => feedback("Could not copy voice ID.", true)));
            const remove = document.createElement("button");
            remove.type = "button";
            remove.textContent = "Delete";
            remove.addEventListener("click", async () => {
                if (!window.confirm(`Delete saved voice "${voice.name}"?`)) return;
                try {
                    await request("/voices/" + encodeURIComponent(voice.id), { method: "DELETE" });
                    await refreshVoices();
                    feedback("Voice deleted.");
                } catch (e) { feedback(e.message, true); }
            });
            tools.append(copy, remove);
            item.append(details, tools);
            list.append(item);
            select.add(new Option(voice.name || voice.id, voice.id));
            emotionSelect.add(new Option(voice.name || voice.id, voice.id));
        }
        if ([...select.options].some((option) => option.value === choice)) select.value = choice;
        if ([...emotionSelect.options].some((option) => option.value === previousEmotionRef)) emotionSelect.value = previousEmotionRef;
        if ((chosenModel.endsWith("-base") || chosenModel === "indextts-2.5") && select.value === "default" && voices.length) {
            select.value = voices[0].id;
        }
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
            if (!destroyed) button.disabled = false;
            refreshStatus().catch(() => {});
        }
    }

    $("[data-action=refresh]").addEventListener("click", () =>
        refreshStatus().catch((e) => feedback(e.message, true)));
    $("[data-action=refresh-voices]").addEventListener("click", () =>
        refreshVoices().catch((e) => feedback(e.message, true)));
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
        refreshVoices().catch((err) => feedback(err.message, true));
    });
    $("[data-selected-model]").addEventListener("change", updateModelDownloadButton);
    $("[data-index-emo-mode]").addEventListener("change", updateEmotionControls);
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

    $("[data-action=clone]").addEventListener("click", (e) => perform(
        e.currentTarget,
        async () => {
            const file = $("[data-clone-file]").files[0];
            if (!file) throw new Error("Select a reference recording.");
            if (!$("[data-clone-consent]").checked) throw new Error("Confirm authorization to use this voice.");
            const form = new FormData();
            form.append("name", $("[data-clone-name]").value.trim());
            form.append("reference_text", $("[data-clone-transcript]").value);
            form.append("consent", "true");
            form.append("audio", file);
            const voice = await (await request("/voices/clone", { method: "POST", body: form })).json();
            await refreshVoices();
            $("[data-test-voice]").value = voice.id;
            activeTab("test");
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
                seed: Number($("[data-test-seed]").value),
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
            const response = await request("/speech", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const blob = await response.blob();
            if (audioURL) URL.revokeObjectURL(audioURL);
            audioURL = URL.createObjectURL(blob);
            const player = $("[data-audio]");
            player.src = audioURL;
            player.hidden = false;
            player.play().catch(() => {});
        }, "Generation complete."));

    refreshModels().catch((e) => feedback(e.message, true));
    refreshDownloadStatus().catch(() => {});
    refreshStatus().catch((e) => feedback(e.message, true));
    timer = window.setInterval(() => {
        if (!destroyed && document.visibilityState !== "hidden") {
            refreshStatus().catch(() => {});
            refreshDownloadStatus().catch(() => {});
        }
    }, 4000);

    activePanel = {
        destroy() {
            destroyed = true;
            if (timer) window.clearInterval(timer);
            if (audioURL) URL.revokeObjectURL(audioURL);
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
