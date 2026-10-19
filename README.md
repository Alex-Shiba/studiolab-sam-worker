# SAM 2.1 Large Vast PyWorker

Publish **this folder alone**, with `worker.py` and `requirements.txt` at the public repository root. No StudioLab source, pictures, model weights, credentials or local caches belong in that repository. No Vast calls or paid deployment were made during development.

## Runtime and template

No custom image: the template uses Vast's own base image, which the platform starts in serverless mode. With `SERVERLESS=true`
(injected by the platform) its supervisor runs `pyworker.sh`, which fetches Vast's official `start_server.sh`. That script clones
`PYWORKER_REPO` at `PYWORKER_REF`, creates the Python 3.10 venv from `requirements.txt`, signs the TLS certificate and runs
`python -m worker`. Our `worker.py` starts `server.py` as a separate process in that venv and owns its termination. No provisioning
script is needed.

The image `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py310-2026-10-13` (Docker Hub, about 8.4 GB) carries CUDA 12.8 and
cuDNN 9, which `onnxruntime-gpu==1.23.2` needs (CUDA 12.x, cuDNN 9:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). The wheel is pinned to 1.23.2 because 1.24.3
was unavailable for Python 3.10. CUDA TF32 is disabled to keep fp32 precision. The server refuses to start when CUDA is not the active
provider, so a library mismatch shows up as `SAM_FATAL` in the worker log, never as silent CPU work.

Template settings:

| Setting | Value |
|---|---|
| Image | `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py310-2026-10-13` |
| Launch mode | the image's default (supervisor); no on-start script |
| `PYWORKER_REPO` | `https://github.com/Alex-Shiba/studiolab-sam-worker` |
| `PYWORKER_REF` | full commit SHA of the published repository |
| `SDK_VERSION` | `1.8.3`, also pinned in requirements; leave `SDK_BRANCH` unset |
| `WORKER_PORT` | `3000` |
| Port mapping | `-p 3000:3000`; Vast injects `VAST_TCP_PORT_3000` and `PUBLIC_IPADDR` |
| `USE_SSL` | `true`; leave `UNSECURED` unset |
| `SAM_CACHE_DIR` | `/workspace/sam-models` on the worker's persistent disk |
| `SAM_LOG_FILE` | `/var/log/portal/sam.log`, shared by model and proxy |
| `SAM_ALLOW_CPU`, `SAM_FORCE_CPU`, `SAM_MODEL_DIR` | unset in deployment |
| Disk | 24 GB: image about 8.4 GB, venv about 1 GB, weights 889,937,943 bytes, logs and headroom (**to measure on the first worker**) |

Only port 3000 is public. The model binds `127.0.0.1:18000`. The disk survives Stopped, but destruction loses it. Missing model files download from the pinned public Hugging Face revision with no token. Both size and SHA-256 are checked every startup; bad cache files move to unique `.bad-*` names. `SAM_MODEL_DIR`, intended for local verification, is strictly read-only and never downloads or quarantines files there.

The model warms up before binding and writes `SAM_READY` only after the socket starts. Fatal startup writes `SAM_FATAL <exception class>`; individual encode failures return 500 without that marker. Startup truncates the old readiness log. The Vast proxy passes requests on in parallel, so one slow download never holds the next request; the model server still runs one encode at a time on the GPU (an asyncio lock), and health checks remain responsive. The model never logs input images or exception messages. `vastai` 1.8.3 logs complete request payloads at DEBUG (`serverless/server/lib/backend.py:724`); the worker globally disables DEBUG after constructing the SDK worker and before running it. A regression test covers this guard.

## Endpoint and workergroup

Set every value explicitly; defaults differ between the UI and API.

| Field | Value | Confirmation |
|---|---:|---|
| `min_load` | 0 | Documented |
| `cold_workers` | 0 | Без холодного worker: остановленная машина застревала с `resources_unavailable`, когда GPU занимал другой арендатор; Vast не заменял её. При пробуждении создаётся новая машина. |
| `max_workers` | 2 | Documented |
| `test_workers` | 0 if accepted, otherwise the lowest accepted value | **UNCONFIRMED minimum and billing; usually a workergroup field, not endpoint** |
| `inactivity_timeout` | 600 | Documented seconds |
| `max_queue_time` | 60 | Documented seconds; handler also uses 60 |
| `target_util` | 0.9 | Documented |
| workergroup `gpu_ram` | 12 GB | **UNCONFIRMED REST units: docs conflict (GB vs MB); verify returned filter** |
| `search_params` | `verified=true rentable=true gpu_ram>=12 cuda_vers>=12.8 dph<=0.30` | See caveats below |

The search selects verified hosts, at least 12 GB VRAM and CUDA 12.8 or later, with a **$0.30/hour ceiling**. This price is a proposed ceiling, **not a confirmed available offer or total-spend cap**. **UNCONFIRMED:** workergroup parsing/units of `gpu_ram`, whether `dph` is an alias there (the digest documents `dph_total`), and whether this exact filter is accepted. If the surface uses `dph_total`, set `dph_total<=0.30`; verify the saved effective search rather than omitting the ceiling. The 12 GB requirement is from the brief; actual model peak VRAM and GPU compatibility (especially Blackwell) remain unmeasured. Template-derived filters can override the query; read back the workergroup configuration before activation.

No claims are made about cold-start time, billable test-worker counts, bandwidth price, worker certificate public-IP SAN, or real GPU feature parity. All remain **UNCONFIRMED**. TLS clients must trust Vast's CA and keep hostname verification; the worker retains Vast's authentication. No account or endpoint credentials are needed by this repository's tests.

## Самоудаление

Дважды 2026-10-19 остановленный после 600 секунд простоя worker застревал:
при пробуждении Vast возвращал `resources_unavailable`, если GPU уже занял другой
арендатор, и не подбирал новую машину. Оба наблюдения относятся к 2026-10-19;
отдельная вторая дата в исходных данных не указана.

Поэтому сервер удаляет свой инстанс после `SAM_SELF_DESTRUCT_IDLE_S` секунд
простоя (по умолчанию 540, то есть 9 минут; `0` отключает самоудаление).
Начало и завершение каждого `/embed`, включая проверочные запросы прокси,
обновляют часы простоя; `/health` не влияет на них. Пока есть запросы в работе,
удаление не начинается. Проверка выполняется каждые 15 секунд.
Используются внедрённые Vast `CONTAINER_ID` и `CONTAINER_API_KEY`; без них
сервер один раз пишет причину пропуска в журнал модели и ничего не вызывает.
Ключ не записывается в журнал. Тайм-аут DELETE — 20 секунд; при сетевой ошибке
или 5xx допускаются три повтора с интервалом 30 секунд, при 4xx повторов нет.
После отправки DELETE новые запросы не отменяют удаление; сервер обслуживает
их до завершения контейнера. У endpoint сохраняется `cold_workers 0`.

Проверка на живом endpoint: разбудить worker, дождаться завершения запросов,
подождать 9–10 минут без новых запросов и убедиться, что инстанс исчез из аккаунта.
Следующий запрос маршрута должен запустить подбор новой машины.
Локальные тесты этого поведения используют подменённые часы и HTTP, без сети:
`python -m pytest -q tests/test_self_destruct.py`. Если pytest отсутствует,
доступен запуск через стандартную библиотеку:
`python -m unittest discover -s tests -p test_self_destruct.py`.

## Contract

`GET /health` returns 200 after warm-up. `POST /embed` accepts JSON:

```json
{"image":"BASE64_PNG","model":"sam2.1-large","revision":"3c23431f721e69cae82dbfd0c28fd692cc714021"}
```

The PNG must be exactly 1024 by 1024, RGB, 8 bits/channel, with no alpha. Inputs are already stretched by StudioLab. Arithmetic matches `src/select/engine.ts`: RGB / 255, ImageNet mean `[.485,.456,.406]` and std `[.229,.224,.225]`, float64 arithmetic then CHW float32 `[1,3,1024,1024]`, input `pixel_values`.

Success: `application/octet-stream`, exactly **8,388,616 bytes**. First `SAMF`, then little-endian uint32 `1`, then little-endian C-order fp16 tensors `image_embeddings.0 [1,32,256,256]`, `.1 [1,64,128,128]`, `.2 [1,256,64,64]`. No response generator is configured, so Vast passes binary through. Input size/mode/format errors return 400 `{"error":"bad_image"}`, model/revision errors 400 `{"error":"bad_model"}`, encode errors 500 `{"error":"encode_failed"}`. The model accepts up to 8 MiB of JSON.

Vast's public `/embed` accepts its normal authenticated envelope with this object under `payload`, not a naked model request. Workload/cost is always 100; one FIFO handler has one synthetic RGB PNG benchmark, two runs plus SDK warm-up. Error patterns are only `SAM_FATAL`; no generic traceback match.

## Local CPU acceptance

From this folder on Windows:

```powershell
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-test.txt
.venv/Scripts/python.exe -m pytest --basetemp=.cache/pytest -q
.venv/Scripts/python.exe tools/parity.py --studio .. --models ../data/models/sam2.1-large
```

The venv was made with Python 3.10.6. Tests force `SAM_ALLOW_CPU=1` and `SAM_FORCE_CPU=1`, read only the two named local encoder files, start the real server on 18000, and test the binary contract. `SAM_TEST_MODEL_DIR` can point tests at another verified local copy. No tests contact Hugging Face or Vast; downloader tests mock the network.

Parity uses StudioLab's own `ServerSelector`, `sharp`, and `onnxruntime-node` from the supplied worktree. Only public `amelie.jpg` and `blade.jpg` samples are used. The tool writes RGB8, normalized inputs, and Node features into ignored `.cache/parity`, captures Python fp32 in-process without adding a debug route, and compares the actual HTTP fp16 response. It reports raw elementwise maximum relative error, including near-zero values; it fails if any comparison reaches 1e-3. Results are in [PARITY.md](PARITY.md). GPU parity is a separate unperformed check.

Measured locally: **26 tests passed** in 53.10 seconds. Python and Node inputs and outputs are bit-identical for both frames, including fp16-to-fp16 comparison (all absolute/relative errors zero). The stricter fp16-wire versus original fp32 comparison fails the 1e-3 elementwise relative threshold: near-zero values produce up to 1.0 relative error even when quantizing Node's own output. The largest tensor-normalized infinity error is 0.00044800665. The tool exits 1 rather than treating this alternative metric as an approved replacement. The conductor must settle that acceptance ambiguity. The Windows test run also emitted a third-party `pyreadline3` console-cleanup warning at interpreter shutdown; pytest exited 0. `pip check` and the Node tool's ESLint check passed.

## Sources and needs

The Vast facts above come from the supplied docs/examples/critic digests, especially:

- https://github.com/vast-ai/pyworker/blob/main/start_server.sh
- https://github.com/vast-ai/vast-cli/tree/master/vastai/serverless
- https://docs.vast.ai/guides/serverless/creating-new-pyworkers
- https://docs.vast.ai/guides/serverless/workergroup-parameters
- https://docs.vast.ai/guides/serverless/serverless-parameters

**needs:** published commit; confirmation that the base image starts the PyWorker and that CUDA is active; effective port mapping/platform variables; disk sizing; GPU VRAM/driver/architecture validation and CUDA parity; `gpu_ram` units and exact search filter syntax; `dph` vs `dph_total` and availability below the proposed ceiling; minimum accepted `test_workers` and its billing; actual cold-start time, bandwidth price and TLS certificate SAN; resolution of the fp16 acceptance threshold; conductor's Opus review (unavailable in this executor). The complete machine-readable list is [needs.json](needs.json). No live endpoint is created by this project.
