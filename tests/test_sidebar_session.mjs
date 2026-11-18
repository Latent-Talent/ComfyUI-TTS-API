"""Pure Node regression checks for sidebar session persistence without ComfyUI.

Tests reuse the real sidebar module code, with only the ComfyUI imports stubbed.
"""
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(join(process.cwd(), "web/tts_sidebar.js"), "utf8")
    .replace(/^import .*;\s*$/gm, "");

function build(storage) {
    const window = {
        sessionStorage: {
            getItem: (name) => storage.get(name) ?? null,
            setItem: (name, value) => storage.set(name, String(value)),
            removeItem: (name) => storage.delete(name),
        },
    };
    const context = vm.createContext({
        window,
        app: {registerExtension: () => {}},
        api: {fetchApi: () => {throw new Error("Unexpected request");}},
    });
    vm.runInContext(source + `
        globalThis.__sessionTest = {
            rememberComfyuiSession,
            storedSidebarSession,
            clearSidebarSession,
            SESSION_FIELDS,
            SIDEBAR_SESSION_KEY,
        };
    `, context);
    return context.__sessionTest;
}

test("test inputs and voice selection survive the sidebar session, not credentials", () => {
    const fields = build(new Map()).SESSION_FIELDS;
    for (const required of [
        "test-model", "test-language", "test-voice", "test-text",
        "test-delivery", "test-seed", "test-steps",
        "index-emo-mode", "index-emo-reference", "index-emo-strength",
        "index-duration", "selected-model",
    ]) {
        assert.ok(fields.includes(required), required);
    }
    for (const excluded of [
        "api-key", "clone-file", "clone-consent", "index-emo-file",
        "index-emo-file-consent",
    ]) {
        assert.ok(!fields.includes(excluded), excluded);
    }
});

test("ComfyUI restart clears stored tab state, TTS worker restart does not", () => {
    const storage = new Map();
    const client = build(storage);
    const key = client.SIDEBAR_SESSION_KEY;
    const snapshot = {comfyuiSession: "boot-a", tab: "test",
        fields: {"test-text": "Unsent speech", "test-model": "voxcpm2"}};
    storage.set(key, JSON.stringify(snapshot));

    assert.equal(client.rememberComfyuiSession("boot-a"), false);
    assert.equal(client.rememberComfyuiSession("boot-a"), false);
    assert.equal(client.storedSidebarSession().fields["test-text"], "Unsent speech");

    assert.equal(client.rememberComfyuiSession("boot-b"), true);
    assert.equal(client.storedSidebarSession(), null);
    assert.equal(storage.has(key), false);
});

test("browser refresh restores state only for the same ComfyUI process", () => {
    const storage = new Map();
    const previous = build(storage);
    const key = previous.SIDEBAR_SESSION_KEY;
    storage.set(key, JSON.stringify({comfyuiSession: "boot-a", tab: "test",
        fields: {"test-text": "Still here"}}));

    const reloaded = build(storage);
    assert.equal(reloaded.rememberComfyuiSession("boot-a"), false);
    assert.equal(reloaded.storedSidebarSession().tab, "test");
    const nextRestart = build(storage);
    assert.equal(nextRestart.rememberComfyuiSession("boot-c"), true);
    assert.equal(nextRestart.storedSidebarSession(), null);
});

test("corrupt or disabled sessionStorage does not throw", () => {
    const storage = new Map();
    const client = build(storage);
    storage.set(client.SIDEBAR_SESSION_KEY, "{garbage");
    assert.equal(client.storedSidebarSession(), null);
    assert.equal(client.rememberComfyuiSession("boot-a"), false);
    client.clearSidebarSession();
    assert.equal(storage.has(client.SIDEBAR_SESSION_KEY), false);
});
