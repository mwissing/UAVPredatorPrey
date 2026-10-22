"""Read scalar summaries from TensorBoard event files with the standard library only.

The host has no numpy/torch/tensorboard (Docker-only runtime), so this decodes
the TFRecord framing (uint64 length, uint32 CRC, payload, uint32 CRC) and the
Event protobuf directly: field 2 = step, field 5 = Summary; Summary field 1 =
Value; Value field 1 = tag, field 2 = simple_value (float32).

Usage: python3 tb_scalars.py <events-file> [tag-substring ...]

Copied from artifacts/isaac/diagnostics/tb_scalars.py so the recipe is self-contained in the repo.
"""

from __future__ import annotations

import struct
import sys
from collections import defaultdict


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, value


def read_scalars(path: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    data = open(path, "rb").read()
    pos = 0
    while pos + 12 <= len(data):
        (length,) = struct.unpack_from("<Q", data, pos)
        payload = data[pos + 12:pos + 12 + length]
        pos += 12 + length + 4
        step, summary = 0, None
        for field, _wire, value in _fields(payload):
            if field == 2:
                step = value
            elif field == 5:
                summary = value
        if summary is None:
            continue
        for field, _wire, value_msg in _fields(summary):
            if field != 1:
                continue
            tag = scalar = None
            for vfield, vwire, vvalue in _fields(value_msg):
                if vfield == 1:
                    tag = vvalue.decode("utf-8", "replace")
                elif vfield == 2 and vwire == 5:
                    (scalar,) = struct.unpack("<f", vvalue)
            if tag is not None and scalar is not None:
                series[tag].append((step, scalar))
    return series


if __name__ == "__main__":
    scalars = read_scalars(sys.argv[1])
    filters = sys.argv[2:]
    for tag in sorted(scalars):
        if filters and not any(f in tag for f in filters):
            continue
        points = scalars[tag]
        print(f"{tag}\tn={len(points)}\tfirst={points[0]}\tlast={points[-1]}")
