"""Vast proxy configuration; TLS remains owned by start_server.sh."""

import base64
import io
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys

from PIL import Image
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from contract import MODEL, REVISION


def benchmark_payload():
    stream = io.BytesIO()
    Image.new("RGB", (1024, 1024), (117, 128, 139)).save(stream, format="PNG")
    return {"image": base64.b64encode(stream.getvalue()).decode("ascii"), "model": MODEL, "revision": REVISION}


def build_config():
    return WorkerConfig(
        model_server_url="http://127.0.0.1", model_server_port=18000,
        model_log_file=os.environ.get("SAM_LOG_FILE", "/var/log/portal/sam.log"),
        model_healthcheck_url="/health",
        handlers=[HandlerConfig(
            route="/embed", allow_parallel_requests=False, max_queue_time=60.0,
            workload_calculator=lambda payload: 100.0,
            benchmark_config=BenchmarkConfig(generator=benchmark_payload, runs=2, concurrency=1),
        )],
        log_action_config=LogActionConfig(on_load=["SAM_READY"], on_error=["SAM_FATAL"]),
    )


def main():
    # start_server.sh has already cloned the repo, installed deps and set up TLS.
    # Own the separate model process so template provisioning cannot race the clone.
    log = Path(os.environ.get("SAM_LOG_FILE", "/var/log/portal/sam.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("server.py"))])

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        proxy = Worker(build_config())
        # vastai 1.8.3 sets DEBUG in Worker.__init__ and logs full image payloads.
        # Disable DEBUG globally after initialization, before any proxy work starts.
        logging.disable(logging.DEBUG)
        proxy.run()
    finally:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


if __name__ == "__main__":
    main()
