"""Entry point for the ComfyUI-managed TTS API worker.

The ComfyUI Manager can restart via os.execv(), retaining the parent's PID.
An inherited CLOEXEC liveness pipe therefore detects a *replaced* parent
even though os.getppid() remains unchanged. The guardian starts before
importing the heavy TTS server and exits when that pipe closes.

The normal standalone start.sh continues to launch uvicorn directly and
does not require a parent liveness pipe.
"""
from __future__ import annotations

import argparse
import os
import threading


def _watch_parent(fd: int) -> None:
    try:
        # No data is sent. EOF signals that the original parent exited or
        # exec-restarted; neither a PID check nor a TCP probe can catch both.
        while os.read(fd, 1):
            pass
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    os._exit(0)


def start_parent_watchdog() -> bool:
    value = os.environ.get("TTS_PARENT_PIPE_FD", "")
    if not value:
        return False
    try:
        fd = int(value)
        if fd < 3:
            raise ValueError()
        os.fstat(fd)
    except (OSError, ValueError):
        raise SystemExit("Invalid managed TTS parent watchdog descriptor")
    threading.Thread(
        target=_watch_parent, args=(fd,), name="TTS-Parent-Watchdog",
        daemon=True,
    ).start()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI-managed TTS API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8025)
    args = parser.parse_args()
    start_parent_watchdog()
    # Deliberately import Uvicorn and the TTS server after the guardian is
    # established; model initialization cannot outlive an exec restart.
    import uvicorn
    uvicorn.run("tts_api.server:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
