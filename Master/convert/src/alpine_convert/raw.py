from __future__ import annotations
import json, struct
from pathlib import Path

MAGIC = b"ALPN2745RAW"
FILE_PREFIX = struct.Struct("<11sBI")
EVENT_HEADER = struct.Struct("<QQII")
CHANNEL_HEADER = struct.Struct("<HI")

class RawFormatError(RuntimeError):
    pass

def read_header(f):
    b = f.read(FILE_PREFIX.size)
    if len(b) != FILE_PREFIX.size:
        raise RawFormatError("Truncated RAW header")
    magic, version, json_len = FILE_PREFIX.unpack(b)
    if magic != MAGIC:
        raise RawFormatError(f"Unexpected RAW magic {magic!r}")
    meta_bytes = f.read(json_len)
    if len(meta_bytes) != json_len:
        raise RawFormatError("Truncated RAW metadata")
    meta = json.loads(meta_bytes)
    if meta.get("format") != "ALPINE-DT2745-RAW":
        raise RawFormatError(f"Unexpected RAW format {meta.get('format')!r}")
    if meta.get("byteOrder") != "little" or meta.get("sampleType") != "uint16":
        raise RawFormatError("Only little-endian uint16 ALPINE RAW is supported")
    meta["_file_format_version"] = version
    meta["_data_offset"] = FILE_PREFIX.size + json_len
    return meta

def iter_events(path: Path):
    f = path.open("rb")
    meta = read_header(f)
    try:
        while True:
            b = f.read(EVENT_HEADER.size)
            if not b:
                break
            if len(b) != EVENT_HEADER.size:
                raise RawFormatError("Truncated event header")
            event, ticks, trigger_id, n_channels = EVENT_HEADER.unpack(b)
            channels = []
            for _ in range(n_channels):
                h = f.read(CHANNEL_HEADER.size)
                if len(h) != CHANNEL_HEADER.size:
                    raise RawFormatError("Truncated channel header")
                ch, n_samples = CHANNEL_HEADER.unpack(h)
                sb = f.read(n_samples * 2)
                if len(sb) != n_samples * 2:
                    raise RawFormatError(f"Truncated waveform event={event} channel={ch}")
                samples = struct.unpack(f"<{n_samples}H", sb)
                channels.append((ch, samples))
            yield meta, event, ticks, trigger_id, channels
    finally:
        f.close()

def inspect_raw(path: Path, max_events=3):
    out = {"metadata": None, "events": []}
    for i, (meta, event, ticks, trigger, channels) in enumerate(iter_events(path)):
        if out["metadata"] is None:
            out["metadata"] = meta
        if i >= max_events:
            break
        out["events"].append({
            "event": event,
            "timestamp_ticks": ticks,
            "trigger_id": trigger,
            "channels": [{"channel": c, "n_samples": len(s)} for c, s in channels],
        })
    return out
