import asyncio
import base64
import hashlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
import numpy as np
from PIL import Image
import pytest

from contract import HEADER, MODEL, REVISION, SHAPES, WIRE_SIZE
import server
from worker import benchmark_payload, build_config

ROOT = Path(__file__).resolve().parents[1]


def png(mode="RGB", size=(1024, 1024), fmt="PNG"):
    stream = io.BytesIO()
    Image.new(mode, size).save(stream, format=fmt)
    return base64.b64encode(stream.getvalue()).decode()


def payload(image=None):
    return {"model": MODEL, "revision": REVISION, "image": image if image is not None else png()}


class FakeEncoder:
    failed = False

    def encode(self, rgb):
        assert rgb.shape == (1024, 1024, 3) and rgb.dtype == np.uint8
        if self.failed:
            self.failed = False
            raise RuntimeError("private image data must never be logged")
        return [np.full(shape, i + .25, np.float32) for i, shape in enumerate(SHAPES)]


@pytest.fixture
async def client():
    encoder = FakeEncoder()
    connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
    async with TestClient(TestServer(server.make_app(encoder)), connector=connector) as client:
        yield client, encoder


async def test_wire_order_and_recovery(client, capsys):
    http, encoder = client
    encoder.failed = True
    response = await http.post("/embed", json=payload())
    assert response.status == 500 and await response.json() == {"error": "encode_failed"}
    assert (await http.get("/health")).status == 200
    response = await http.post("/embed", json=payload())
    body = await response.read()
    assert response.status == 200 and response.headers["Content-Type"] == "application/octet-stream"
    assert len(body) == WIRE_SIZE and body[:8] == HEADER
    offset = 8
    for i, shape in enumerate(SHAPES):
        size = int(np.prod(shape)) * 2
        assert np.all(np.frombuffer(body[offset:offset + size], "<f2") == i + .25)
        offset += size
    assert "private" not in capsys.readouterr().out


@pytest.mark.parametrize("image", [png(size=(1023, 1024)), png(mode="RGBA"), png(mode="L"), png(mode="P"), png(fmt="JPEG"), "!", "", None, 123],
                         ids=["size", "rgba", "gray", "palette", "jpeg", "base64", "empty", "null", "number"])
async def test_invalid_image(client, image):
    http, _ = client
    request = payload()
    request["image"] = image
    response = await http.post("/embed", json=request)
    assert response.status == 400 and await response.json() == {"error": "bad_image"}


def test_reject_rgb16():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    # Valid RGB16 PNG, which Pillow presents as mode RGB.
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1024, 1024, 16, 2, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress((b"\0" + bytes(1024 * 6)) * 1024)) + chunk(b"IEND", b"")
    assert Image.open(io.BytesIO(raw)).mode == "RGB"
    with pytest.raises(ValueError):
        server.decode_image(base64.b64encode(raw).decode())


def test_reject_rgb_transparency():
    stream = io.BytesIO()
    Image.new("RGB", (1024, 1024)).save(stream, format="PNG", transparency=(0, 0, 0))
    with pytest.raises(ValueError):
        server.decode_image(base64.b64encode(stream.getvalue()).decode())


@pytest.mark.parametrize("field,value", [("model", "sam2.1-tiny"), ("revision", "wrong"), ("model", None)])
async def test_bad_model(client, field, value):
    http, _ = client
    request = payload()
    request[field] = value
    response = await http.post("/embed", json=request)
    assert response.status == 400 and await response.json() == {"error": "bad_model"}


async def test_invalid_json(client):
    http, _ = client
    response = await http.post("/embed", data=b"{")
    assert response.status == 400 and await response.json() == {"error": "bad_image"}


def test_config():
    config = build_config()
    assert config.model_server_url == "http://127.0.0.1" and config.model_server_port == 18000
    assert config.model_healthcheck_url == "/health"
    assert config.log_action_config.on_load == ["SAM_READY"]
    assert config.log_action_config.on_error == ["SAM_FATAL"]
    assert len(config.handlers) == 1
    handler = config.handlers[0]
    assert handler.route == "/embed" and handler.allow_parallel_requests is False
    assert handler.max_queue_time == 60 and handler.workload_calculator({}) == 100
    assert handler.response_generator is None
    assert handler.benchmark_config is not None
    sample = handler.benchmark_config.generator()
    assert sample["model"] == MODEL and sample["revision"] == REVISION
    assert server.decode_image(sample["image"]).shape == (1024, 1024, 3)
    assert len(sample["image"]) < 10000


def test_all_normalization_values():
    rgb = np.tile(np.arange(256, dtype=np.uint8)[:, None, None], (1, 1, 3))
    actual = server.normalize(rgb)
    for c, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
        expected = np.array([(i / 255 - mean) / std for i in range(256)], dtype=np.float32)
        np.testing.assert_array_equal(actual[0, c, :, 0], expected)


def test_cache_download_and_quarantine(tmp_path, monkeypatch):
    contents = b"verified-model"
    digest = hashlib.sha256(contents).hexdigest()
    monkeypatch.delenv("SAM_MODEL_DIR", raising=False)
    monkeypatch.setenv("SAM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(server, "FILES", {"vision_encoder.onnx": (len(contents), digest)})
    target = tmp_path / "vision_encoder.onnx"
    target.write_bytes(b"wrong")
    calls = []

    def download(url, timeout):
        calls.append(url)
        assert REVISION in url and timeout == 120
        return io.BytesIO(contents)

    monkeypatch.setattr(server.urllib.request, "urlopen", download)
    assert server.model_path().read_bytes() == contents
    assert len(list(tmp_path.glob("*.bad-*"))) == 1
    assert server.model_path() == target and len(calls) == 1
    target.write_bytes(b"x" * len(contents))
    monkeypatch.setenv("SAM_MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="invalid_local_model"):
        server.model_path()
    assert target.read_bytes() == b"x" * len(contents) and len(calls) == 1


def test_corrupt_download_not_installed(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM_MODEL_DIR", raising=False)
    monkeypatch.setenv("SAM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(server, "FILES", {"vision_encoder.onnx": (2, hashlib.sha256(b"ok").hexdigest())})
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"no"))
    with pytest.raises(RuntimeError, match="invalid_download"):
        server.model_path()
    assert not (tmp_path / "vision_encoder.onnx").exists()
    assert len(list(tmp_path.glob("*.bad-*"))) == 1


def test_cuda_required(monkeypatch):
    monkeypatch.delenv("SAM_ALLOW_CPU", raising=False)
    monkeypatch.setattr(server.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="cuda_unavailable"):
        server.Encoder()


def test_cuda_silent_fallback_rejected(monkeypatch):
    monkeypatch.delenv("SAM_ALLOW_CPU", raising=False)
    monkeypatch.setattr(server.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(server, "model_path", lambda: Path("unused"))

    class FallbackSession:
        def __init__(self, *args, **kwargs):
            pass

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(server.ort, "InferenceSession", FallbackSession)
    with pytest.raises(RuntimeError, match="cuda_inactive"):
        server.Encoder()


def test_fatal_start_marker(tmp_path):
    log = tmp_path / "fatal.log"
    env = {**os.environ, "SAM_ALLOW_CPU": "1", "SAM_FORCE_CPU": "1", "SAM_MODEL_DIR": str(tmp_path), "SAM_LOG_FILE": str(log)}
    result = subprocess.run([sys.executable, str(ROOT / "server.py")], env=env, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert log.read_text().startswith("SAM_FATAL ") and "SAM_READY" not in log.read_text()


async def test_real_cpu_server(tmp_path):
    models = os.environ.get("SAM_TEST_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "data" / "models" / "sam2.1-large"))
    # Missing model files fail rather than silently skipping the acceptance test.
    assert (Path(models) / "vision_encoder.onnx").is_file()
    log = tmp_path / "sam.log"
    env = {**os.environ, "SAM_ALLOW_CPU": "1", "SAM_FORCE_CPU": "1", "SAM_MODEL_DIR": models, "SAM_LOG_FILE": str(log)}
    with (tmp_path / "process.log").open("wb") as output:
        process = subprocess.Popen([sys.executable, str(ROOT / "server.py")], env=env, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 600
            while not log.exists() or "SAM_READY" not in log.read_text():
                assert process.poll() is None, "Server exited before readiness"
                assert time.monotonic() < deadline, "Server readiness timeout"
                await asyncio.sleep(.5)
            assert "SAM_FATAL" not in log.read_text()
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=connector) as http:
                async with http.get("http://127.0.0.1:18000/health") as response:
                    assert response.status == 200
                async with http.post("http://127.0.0.1:18000/embed", json=benchmark_payload()) as response:
                    body = await response.read()
                    assert response.status == 200
                    assert response.headers["Content-Type"] == "application/octet-stream"
                    assert len(body) == WIRE_SIZE and body[:8] == HEADER
                    assert np.isfinite(np.frombuffer(body[8:], "<f2")).all()
                async with http.post("http://127.0.0.1:18000/embed", json=payload("!")) as response:
                    assert response.status == 400 and await response.json() == {"error": "bad_image"}
                bad_model = payload()
                bad_model["model"] = "wrong"
                async with http.post("http://127.0.0.1:18000/embed", json=bad_model) as response:
                    assert response.status == 400 and await response.json() == {"error": "bad_model"}
        finally:
            process.terminate()
            await asyncio.to_thread(process.wait, 30)
