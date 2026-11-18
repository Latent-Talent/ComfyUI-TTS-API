"""IndexTTS 2.5 inference worker (runs ONLY in its official Python 3.11 venv).

This subprocess is isolated from the Python 3.12 TTS HTTP API. It connects
to a private Unix socket in a mode-0700 temporary directory. No API endpoint,
port, speech text, or model imports are exposed to the parent ComfyUI process.

IndexTTS upstream prints source sentences even at low verbosity. Its stdout
and stderr are kept in a separate raw diagnostic file, never mirrored into the
sanitized ComfyUI console.
"""
from __future__ import annotations

import json
import ctypes
import signal
import os
from pathlib import Path
import random
import socket
import sys
import time


def _recv(reader):
    raw = reader.readline(2 * 1024 * 1024)
    if not raw or not raw.endswith(b"\n"):
        raise ConnectionError("Incomplete private IndexTTS operation")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid private IndexTTS request")
    return data


def _send(connection: socket.socket, item: dict) -> None:
    connection.sendall((json.dumps(item, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii"))


def main() -> int:
    # When ComfyUI force-stops the managed TTS API, IndexTTS must not remain
    # orphaned on the GPU. On Linux/WSL, request SIGTERM when its parent dies.
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None)
            libc.prctl(1, signal.SIGTERM, 0, 0, 0)
        except (AttributeError, OSError):
            pass
    expected_parent = int(os.environ.get("TTS_INDEX_PARENT_PID", "0"))
    if expected_parent and os.getppid() != expected_parent:
        return 1
    source = Path(os.environ["TTS_INDEX_SOURCE"]).resolve()
    checkpoints = Path(os.environ["TTS_INDEX_CHECKPOINTS"]).resolve()
    socket_path = os.environ["TTS_INDEX_SOCKET"]
    token = os.environ["TTS_INDEX_TOKEN"]
    outputs = Path(os.environ["TTS_INDEX_OUTPUTS"]).resolve()
    if not source.is_dir() or not (checkpoints / "config.yaml").is_file():
        return 2
    sys.path.insert(0, str(source))

    # Only the parent-owned TTS process can connect through this private socket.
    with socket.socket(socket.AF_UNIX) as connection:
        for attempt in range(100):
            try:
                connection.connect(socket_path)
                break
            except OSError:
                time.sleep(0.1)
        else:
            return 3
        reader = connection.makefile("rb")
        _send(connection, {"event": "hello", "token": token})
        try:
            import torch
            from .runtime import apply_cpu_threads, quantize_int8, require_int8_backend, resolve_device
            from indextts.infer_v2_5 import IndexTTS2
            device = resolve_device(os.environ.get("TTS_INDEX_DEVICE", "auto"), torch)
            precision = os.environ.get("TTS_INDEX_PRECISION", "native")
            if precision not in ("native", "int8"):
                raise ValueError("Unsupported IndexTTS precision")
            apply_cpu_threads(torch, int(os.environ.get("TTS_INDEX_CPU_THREADS", "0")), device)
            if precision == "int8":
                require_int8_backend()
            model = IndexTTS2(
                cfg_path=str(checkpoints / "config.yaml"),
                model_dir=str(checkpoints),
                device=device,
                use_bf16=bool(device.startswith("cuda") and torch.cuda.is_bf16_supported()),
                use_cuda_kernel=False, use_deepspeed=False, use_accel=False,
                use_torch_compile=False, use_qwen_emo=True,
            )
            converted = 0
            if precision == "int8":
                # Quantize the language model only; leave speech decoder, audio
                # codec, BigVGAN and speaker analysis in their native precision.
                converted = quantize_int8(model.gpt, scope="index", torch=torch)
            _send(connection, {"event": "ready", "device": device,
                               "quantized_layers": converted})
        except BaseException as exc:
            _send(connection, {"event": "failed", "error_type": type(exc).__name__})
            return 4
        while True:
            try:
                payload = _recv(reader)
            except (ConnectionError, OSError):
                return 0
            if payload.get("op") == "stop":
                return 0
            if payload.get("op") != "generate":
                _send(connection, {"event": "failed", "error_type": "ValueError"})
                continue
            try:
                # Parent controls output naming and only accepts its own
                # per-generation file under the mode-0700 private directory.
                name = payload.get("output")
                if not isinstance(name, str) or len(name) != 36 or not name.endswith(".wav"):
                    raise ValueError("Invalid output name")
                output_path = (outputs / name).resolve()
                if output_path.parent != outputs:
                    raise ValueError("Invalid output destination")

                seed = payload.get("seed")
                if seed is not None:
                    random.seed(int(seed))
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))
                mode = payload["emotion_mode"]
                options = {
                    "spk_audio_prompt": payload["reference"],
                    "text": payload["text"],
                    "lang": payload["language"],
                    "output_path": str(output_path),
                    "emo_alpha": float(payload["emotion_strength"]),
                    "use_random": False,
                    "verbose": False,
                    "duration_factor": float(payload["duration_factor"]),
                }
                if mode == "vector":
                    options["emo_vector"] = payload["emotion_vector"]
                elif mode == "text":
                    options["use_emo_text"] = True
                    options["emo_text"] = payload["instructions"]
                elif mode == "auto":
                    options["use_emo_text"] = True
                elif mode == "reference":
                    options["emo_audio_prompt"] = payload["emotion_reference"]
                else:
                    options["emo_vector"] = [0.0] * 8
                result = model.infer(**options)
                if not output_path.is_file() or output_path.stat().st_size < 44:
                    raise RuntimeError("IndexTTS generated no WAV output")
                _send(connection, {"event": "completed", "bytes": output_path.stat().st_size})
            except BaseException as exc:
                # Upstream errors can quote the entire spoken text. Only the
                # class name crosses into the ComfyUI process/console.
                _send(connection, {"event": "failed", "error_type": type(exc).__name__})


if __name__ == "__main__":
    raise SystemExit(main())
