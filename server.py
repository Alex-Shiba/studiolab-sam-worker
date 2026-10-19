"""Loopback SAM encoder. Images and exception messages are never logged."""

import asyncio
import base64
import hashlib
import io
import os
from pathlib import Path
import struct
import uuid
import urllib.request

from aiohttp import web
import numpy as np
from PIL import Image
import onnxruntime as ort

from contract import HEADER, MODEL, NAMES, REVISION, SHAPES

FILES = {
    "vision_encoder.onnx": (1_350_487, "ad3a8161c6bd6d8beea811300c60e20d626035feaa3c19622ace27b22e25bec8"),
    "vision_encoder.onnx_data": (888_587_456, "88c501a65b6aaeac99b65a275f3f559727404283930d107d499fda8714ba7e5a"),
}


def valid_file(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest() == digest


def model_path():
    # An explicit local source is read-only, including on checksum failure.
    local = os.environ.get("SAM_MODEL_DIR")
    root = Path(local or os.environ.get("SAM_CACHE_DIR", "models"))
    if not local:
        root.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        path = root / name
        if valid_file(path, size, digest):
            continue
        if local:
            raise RuntimeError("invalid_local_model")
        if path.exists():
            path.rename(root / f"{name}.bad-{uuid.uuid4().hex}")
        temporary = root / f"{name}.part-{uuid.uuid4().hex}"
        try:
            url = f"https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/{REVISION}/onnx/{name}"
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(chunk)
            if not valid_file(temporary, size, digest):
                temporary.rename(root / f"{name}.bad-{uuid.uuid4().hex}")
                raise RuntimeError("invalid_download")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return root / "vision_encoder.onnx"


def decode_image(encoded):
    if not isinstance(encoded, str):
        raise ValueError("bad_image")
    raw = base64.b64decode(encoded, validate=True)
    # Check PNG IHDR explicitly: Pillow can downconvert 16-bit RGB to RGB8.
    if (len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n"
            or raw[8:16] != b"\x00\x00\x00\x0dIHDR"
            or struct.unpack(">IIBB", raw[16:26]) != (1024, 1024, 8, 2)):
        raise ValueError("bad_image")
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format != "PNG" or image.size != (1024, 1024) or image.mode != "RGB"
                or "transparency" in image.info or getattr(image, "n_frames", 1) != 1):
            raise ValueError("bad_image")
        image.load()
        return np.array(image, dtype=np.uint8)


def normalize(rgb):
    # JS Number arithmetic is float64, with a single Float32Array assignment.
    values = rgb.astype(np.float64) / 255
    values = (values - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=np.float32)


def pack_features(outputs):
    if len(outputs) != 3:
        raise ValueError("invalid_features")
    chunks = [HEADER]
    for output, shape in zip(outputs, SHAPES):
        if output.shape != shape or output.dtype != np.float32 or not np.isfinite(output).all():
            raise ValueError("invalid_features")
        with np.errstate(over="raise", invalid="raise"):
            chunks.append(output.astype("<f2").tobytes(order="C"))
    return b"".join(chunks)


class Encoder:
    def __init__(self):
        allow_cpu = os.environ.get("SAM_ALLOW_CPU") == "1"
        providers = ort.get_available_providers()
        if not allow_cpu and "CUDAExecutionProvider" not in providers:
            raise RuntimeError("cuda_unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("SAM_THREADS", "2"))
        # Disable TF32 to preserve the pinned fp32 encoder's precision on CUDA.
        selected = [("CUDAExecutionProvider", {"use_tf32": "0"})] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
        if allow_cpu and os.environ.get("SAM_FORCE_CPU") == "1":
            selected = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path()), sess_options=options, providers=selected)
        self.session.disable_fallback()
        if not allow_cpu and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("cuda_inactive")

    def encode(self, rgb):
        return self.session.run(list(NAMES), {"pixel_values": normalize(rgb)})


def make_app(encoder):
    app = web.Application(client_max_size=8 * 1024 * 1024)
    lock = asyncio.Lock()

    async def health(request):
        return web.json_response({"status": "ready"})

    async def embed(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "bad_image"}, status=400)
        if not isinstance(payload, dict) or payload.get("model") != MODEL or payload.get("revision") != REVISION:
            return web.json_response({"error": "bad_model"}, status=400)
        try:
            rgb = decode_image(payload.get("image"))
        except Exception:
            return web.json_response({"error": "bad_image"}, status=400)
        try:
            async with lock:
                body = await asyncio.to_thread(lambda: pack_features(encoder.encode(rgb)))
            return web.Response(body=body, content_type="application/octet-stream")
        except Exception:
            # No image, exception text or fatal marker for a per-request failure.
            return web.json_response({"error": "encode_failed"}, status=500)

    app.router.add_get("/health", health)
    app.router.add_post("/embed", embed)
    return app


def log_marker(line):
    path = Path(os.environ.get("SAM_LOG_FILE", "/var/log/portal/sam.log"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(line + "\n")
        log.flush()


async def serve():
    runner = None
    try:
        encoder = await asyncio.to_thread(Encoder)
        await asyncio.to_thread(lambda: pack_features(encoder.encode(np.zeros((1024, 1024, 3), dtype=np.uint8))))
        runner = web.AppRunner(make_app(encoder), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18000).start()
        log_marker("SAM_READY")
        await asyncio.Event().wait()
    except Exception as error:
        log_marker(f"SAM_FATAL {type(error).__name__}")
        raise SystemExit(1) from None
    finally:
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(serve())
