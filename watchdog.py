"""Trusted cloud timer: runs outside the agent's sandbox."""

import time
import modal

app = modal.App("mobilellm-watchdog", include_source=False)


@app.function(image=modal.Image.debian_slim().pip_install("modal>=1.6,<2"),
              serialized=True, cpu=0.125, timeout=86400)
def enforce_deadline(sandbox_id: str, deadline: float):
    while time.time() < deadline:
        if modal.Sandbox.from_id(sandbox_id).poll() is not None:
            return
        time.sleep(min(30, max(0, deadline - time.time())))
    modal.Sandbox.from_id(sandbox_id).terminate(wait=True)
