"""Shared wire constants; no inference or network side effects."""

MODEL = "sam2.1-large"
REVISION = "3c23431f721e69cae82dbfd0c28fd692cc714021"
SHAPES = ((1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64))
NAMES = tuple(f"image_embeddings.{i}" for i in range(3))
HEADER = b"SAMF\x01\x00\x00\x00"
WIRE_SIZE = 8 + 8_388_608
