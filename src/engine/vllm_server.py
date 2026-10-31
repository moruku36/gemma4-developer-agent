"""Lifecycle management for a local vLLM OpenAI-compatible server.

The server runs as a subprocess in its own process group. ``start()``
blocks until ``GET /health`` succeeds (model load on L4 can take minutes),
and ``stop()`` terminates the whole group so no GPU memory leaks into the
next phase of the notebook.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


class VLLMServerError(RuntimeError):
    """Raised when the server cannot start or dies during startup."""


def load_vllm_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class VLLMServer:
    """Spawn, health-check and stop one vLLM server process."""

    def __init__(self, config: dict[str, Any]):
        server_cfg = config.get("server", {})
        model_cfg = config.get("model", {})
        self.host = server_cfg.get("host", "127.0.0.1")
        self.port = int(server_cfg.get("port", 8000))
        self.startup_timeout = float(server_cfg.get("startup_timeout_seconds", 900))
        self.log_file = Path(server_cfg.get("log_file", "artifacts/vllm_server.log"))
        self.model_path = os.environ.get("GEMMA4_MODEL_PATH") or model_cfg.get(
            "path", "models/gemma-4"
        )
        self.served_name = model_cfg.get("served_name", "gemma-4")
        self.model_cfg = model_cfg
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # -- command line ----------------------------------------------------
    def build_command(self) -> list[str]:
        cfg = self.model_cfg
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_path),
            "--served-model-name", self.served_name,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", str(cfg.get("dtype", "bfloat16")),
            "--max-model-len", str(cfg.get("max_model_len", 32768)),
            "--tensor-parallel-size", str(cfg.get("tensor_parallel_size", 1)),
            "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.9)),
            "--seed", str(cfg.get("seed", 42)),
        ]
        if cfg.get("quantization"):
            cmd += ["--quantization", str(cfg["quantization"])]
        if cfg.get("enable_prefix_caching", True):
            cmd += ["--enable-prefix-caching"]
        return cmd

    # -- lifecycle -------------------------------------------------------
    def start(self) -> str:
        """Start the server and block until healthy; return the base URL."""
        try:
            import vllm  # noqa: F401  (presence check only)
        except ImportError as exc:
            raise VLLMServerError(
                "vllm is not installed; run scripts/setup_offline_deps.sh and "
                "install the bundled wheels before starting the server"
            ) from exc
        if not Path(self.model_path).exists():
            raise VLLMServerError(f"model path does not exist: {self.model_path}")
        if self._proc is not None:
            raise VLLMServerError("server already started")

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_file, "ab")
        self._proc = subprocess.Popen(
            self.build_command(),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise VLLMServerError(
                    f"vLLM exited during startup with code {self._proc.returncode}; "
                    f"see {self.log_file}"
                )
            if self._healthy():
                return self.base_url
            time.sleep(5.0)
        self.stop()
        raise VLLMServerError(
            f"vLLM did not become healthy within {self.startup_timeout:.0f}s; "
            f"see {self.log_file}"
        )

    def _healthy(self) -> bool:
        try:
            req = urllib.request.Request(
                f"http://{self.host}:{self.port}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def stop(self) -> None:
        """Terminate the server process group and release the log handle."""
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    self._proc.wait(timeout=30)
                except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    self._proc.wait(timeout=30)
            self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
