"""CPU comparison against StudioLab's actual encoder; writes only under this repo."""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aiohttp
from aiohttp import web
import numpy as np
import onnxruntime as ort

from contract import HEADER, MODEL, REVISION, SHAPES, WIRE_SIZE
from server import Encoder, make_app, normalize, pack_features


def errors(actual, reference):
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denominator = np.abs(reference.astype(np.float64))
    # Report literal elementwise relative error, including near-zero values.
    relative = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator != 0)
    relative[(denominator == 0) & (delta != 0)] = np.inf
    scale = float(denominator.max())
    normalized = float(delta.max()) / scale if scale else float(delta.max())
    return float(delta.max()), float(relative.max()), normalized


async def compare(args, cache):
    os.environ.update(SAM_ALLOW_CPU="1", SAM_FORCE_CPU="1", SAM_MODEL_DIR=str(Path(args.models).resolve()))
    encoder = await asyncio.to_thread(Encoder)
    assert encoder.session.get_providers() == ["CPUExecutionProvider"]
    await asyncio.to_thread(lambda: pack_features(encoder.encode(np.zeros((1024, 1024, 3), np.uint8))))
    # Capture fp32 before transport, using the same server handler and same inference call.
    original_encode = encoder.encode
    captured = []

    def capture(rgb):
        outputs = original_encode(rgb)
        captured[:] = outputs
        return outputs

    encoder.encode = capture
    runner = web.AppRunner(make_app(encoder), access_log=None)
    await runner.setup()
    rows = []
    times = []
    try:
        await web.TCPSite(runner, "127.0.0.1", 18000).start()
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=connector) as client:
            async with client.get("http://127.0.0.1:18000/health") as response:
                assert response.status == 200
            for frame in range(2):
                rgb = np.fromfile(cache / f"{frame}.rgb", np.uint8).reshape(1024, 1024, 3)
                expected_input = np.fromfile(cache / f"{frame}.input.f32", "<f4").reshape(1, 3, 1024, 1024)
                assert np.array_equal(normalize(rgb), expected_input), "Normalization differs from engine.ts"
                payload = {"image": base64.b64encode((cache / f"{frame}.png").read_bytes()).decode(), "model": MODEL, "revision": REVISION}
                start = time.perf_counter()
                async with client.post("http://127.0.0.1:18000/embed", json=payload) as response:
                    assert response.status == 200
                    assert response.headers["Content-Type"] == "application/octet-stream"
                    body = await response.read()
                times.append(time.perf_counter() - start)
                assert len(body) == WIRE_SIZE and body[:8] == HEADER
                offset = 8
                for i, shape in enumerate(SHAPES):
                    reference = np.fromfile(cache / f"{frame}.{i}.f32", "<f4").reshape(shape)
                    size = int(np.prod(shape)) * 2
                    transported = np.frombuffer(body[offset:offset + size], "<f2").astype(np.float32).reshape(shape)
                    offset += size
                    rounded_reference = reference.astype("<f2").astype(np.float32)
                    for label, actual, expected in (
                        ("fp32 Python vs Node", captured[i], reference),
                        ("fp16 Python vs fp16 Node", transported, rounded_reference),
                        ("fp16 wire vs fp32 Node", transported, reference),
                        ("Node fp16 vs Node fp32 (quantization only)", rounded_reference, reference),
                    ):
                        absolute, relative, normalized = errors(actual, expected)
                        rows.append((frame, i, label, absolute, relative, normalized))
                print(f"Server {frame + 1}/2: {times[-1]:.2f} s", flush=True)
    finally:
        await runner.cleanup()
    metadata = json.loads((cache / "reference.json").read_text())
    passed = all(row[4] < 1e-3 for row in rows)
    runtime_passed = all(row[4] < 1e-3 for row in rows if row[2] in ("fp32 Python vs Node", "fp16 Python vs fp16 Node"))
    lines = ["# Проверка совпадения энкодеров", "", f"Результат: {'PASS' if passed else 'FAIL'}; порог максимальной относительной ошибки: < 1e-3.", "",
             f"Сравнение Python/Node при одинаковой точности: {'PASS' if runtime_passed else 'FAIL'}. Общий результат также учитывает потери fp16 относительно fp32.", "",
             f"Python {platform.python_version()}, onnxruntime-gpu {ort.__version__}, CPUExecutionProvider; Node {metadata['node']}, onnxruntime-node {metadata['ort']}, CPU.",
             f"Модель: `{MODEL}`, ревизия `{REVISION}`. Размеры и SHA-256 обоих файлов проверены.", "",
             "Эталон вызывает настоящий `src/select/engine.ts:ServerSelector` с CPU-сессией. Нормализованные входы Python и студии совпадают побитно для обоих кадров.",
             "HTTP: GET /health = 200; POST /embed = 200 application/octet-stream, SAMF/version 1, 8 388 616 байт.", "",
             "Относительная ошибка: max(abs(actual-reference)/abs(reference)) по каждому элементу, без epsilon и без исключения малых значений. Для двух нулей ошибка 0; для отличия от нуля при нулевом эталоне — бесконечность.", "",
             "Дополнительная метрика norm-inf: max(abs(actual-reference))/max(abs(reference)). Она показывает ошибку в масштабе всего тензора, но не заменяет строгий поэлементный порог.", "",
             "| Кадр | Тензор | Сравнение | Max abs | Max rel | Norm-inf |", "|---|---|---|---:|---:|---:|"]
    lines += [f"| {frame + 1} | image_embeddings.{i} | {label} | {absolute:.9g} | {relative:.9g} | {normalized:.9g} |" for frame, i, label, absolute, relative, normalized in rows]
    lines += ["", "Публичные примеры студии:", ""]
    lines += [f"- `{item['frame']}`; SHA-256 `{item['sha256']}`; Node {item['encode_seconds']:.2f} с, HTTP Python {times[i]:.2f} с." for i, item in enumerate(metadata["frames"])]
    lines += ["", "Строки quantization only преобразуют сам эталон Node в fp16 и обратно, без Python-инференса. Превышение порога здесь нельзя исправить в Python-энкодере: fp16 округляет достаточно малые ненулевые значения до нуля. Контракт передачи сохранён без изменений.", "",
              "GPU/CUDA и запуск на Vast не проверялись. Результат CPU не подтверждает численное совпадение CUDA или качество масок после fp16.", ""]
    (ROOT / "PARITY.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studio", default=str(ROOT.parent))
    parser.add_argument("--models", default=str(ROOT.parent / "data" / "models" / "sam2.1-large"))
    parser.add_argument("--reuse-reference", action="store_true")
    args = parser.parse_args()
    cache = ROOT / ".cache" / "parity"
    cache.mkdir(parents=True, exist_ok=True)
    if not args.reuse_reference:
        subprocess.run(["node", str(ROOT / "tools" / "studio-reference.mjs"), args.studio, args.models, str(cache),
                        "web/public/frames/manners/amelie.jpg", "web/public/frames/manners/blade.jpg"], check=True)
    return 0 if asyncio.run(compare(args, cache)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
