# Проверка совпадения энкодеров

Результат: FAIL; порог максимальной относительной ошибки: < 1e-3.

Сравнение Python/Node при одинаковой точности: PASS. Общий результат также учитывает потери fp16 относительно fp32.

Python 3.10.6, onnxruntime-gpu 1.23.2, CPUExecutionProvider; Node v24.21.0, onnxruntime-node 1.24.3, CPU.
Модель: `sam2.1-large`, ревизия `3c23431f721e69cae82dbfd0c28fd692cc714021`. Размеры и SHA-256 обоих файлов проверены.

Эталон вызывает настоящий `src/select/engine.ts:ServerSelector` с CPU-сессией. Нормализованные входы Python и студии совпадают побитно для обоих кадров.
HTTP: GET /health = 200; POST /embed = 200 application/octet-stream, SAMF/version 1, 8 388 616 байт.

Относительная ошибка: max(abs(actual-reference)/abs(reference)) по каждому элементу, без epsilon и без исключения малых значений. Для двух нулей ошибка 0; для отличия от нуля при нулевом эталоне — бесконечность.

Дополнительная метрика norm-inf: max(abs(actual-reference))/max(abs(reference)). Она показывает ошибку в масштабе всего тензора, но не заменяет строгий поэлементный порог.

| Кадр | Тензор | Сравнение | Max abs | Max rel | Norm-inf |
|---|---|---|---:|---:|---:|
| 1 | image_embeddings.0 | fp32 Python vs Node | 0 | 0 | 0 |
| 1 | image_embeddings.0 | fp16 Python vs fp16 Node | 0 | 0 | 0 |
| 1 | image_embeddings.0 | fp16 wire vs fp32 Node | 0.00024408102 | 1 | 0.000246052702 |
| 1 | image_embeddings.0 | Node fp16 vs Node fp32 (quantization only) | 0.00024408102 | 1 | 0.000246052702 |
| 1 | image_embeddings.1 | fp32 Python vs Node | 0 | 0 | 0 |
| 1 | image_embeddings.1 | fp16 Python vs fp16 Node | 0 | 0 | 0 |
| 1 | image_embeddings.1 | fp16 wire vs fp32 Node | 0.00047314167 | 1 | 0.000386391353 |
| 1 | image_embeddings.1 | Node fp16 vs Node fp32 (quantization only) | 0.00047314167 | 1 | 0.000386391353 |
| 1 | image_embeddings.2 | fp32 Python vs Node | 0 | 0 | 0 |
| 1 | image_embeddings.2 | fp16 Python vs fp16 Node | 0 | 0 | 0 |
| 1 | image_embeddings.2 | fp16 wire vs fp32 Node | 0.00373649597 | 0.137320977 | 0.000402698967 |
| 1 | image_embeddings.2 | Node fp16 vs Node fp32 (quantization only) | 0.00373649597 | 0.137320977 | 0.000402698967 |
| 2 | image_embeddings.0 | fp32 Python vs Node | 0 | 0 | 0 |
| 2 | image_embeddings.0 | fp16 Python vs fp16 Node | 0 | 0 | 0 |
| 2 | image_embeddings.0 | fp16 wire vs fp32 Node | 0.000244021416 | 0.329842932 | 0.000244647733 |
| 2 | image_embeddings.0 | Node fp16 vs Node fp32 (quantization only) | 0.000244021416 | 0.329842932 | 0.000244647733 |
| 2 | image_embeddings.1 | fp32 Python vs Node | 0 | 0 | 0 |
| 2 | image_embeddings.1 | fp16 Python vs fp16 Node | 0 | 0 | 0 |
| 2 | image_embeddings.1 | fp16 wire vs fp32 Node | 0.000487446785 | 0.673202614 | 0.000356136274 |
| 2 | image_embeddings.1 | Node fp16 vs Node fp32 (quantization only) | 0.000487446785 | 0.673202614 | 0.000356136274 |
| 2 | image_embeddings.2 | fp32 Python vs Node | 0 | 0 | 0 |
| 2 | image_embeddings.2 | fp16 Python vs fp16 Node | 0 | 0 | 0 |
| 2 | image_embeddings.2 | fp16 wire vs fp32 Node | 0.00374603271 | 0.0666666667 | 0.00044800665 |
| 2 | image_embeddings.2 | Node fp16 vs Node fp32 (quantization only) | 0.00374603271 | 0.0666666667 | 0.00044800665 |

Публичные примеры студии:

- `web/public/frames/manners/amelie.jpg`; SHA-256 `092eef46fe04a6163377e38717a8457cced74a54af3ab5f957f46e3a5727f072`; Node 23.11 с, HTTP Python 21.88 с.
- `web/public/frames/manners/blade.jpg`; SHA-256 `355fcf54e2f053c5f108a270f217be7a9e73e5905b8736af29d218abe164000e`; Node 31.24 с, HTTP Python 22.40 с.

Строки quantization only преобразуют сам эталон Node в fp16 и обратно, без Python-инференса. Превышение порога здесь нельзя исправить в Python-энкодере: fp16 округляет достаточно малые ненулевые значения до нуля. Контракт передачи сохранён без изменений.

GPU/CUDA и запуск на Vast не проверялись. Результат CPU не подтверждает численное совпадение CUDA или качество масок после fp16.
