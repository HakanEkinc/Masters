from __future__ import annotations
import csv, gzip, os
from pathlib import Path
import numpy as np
import uproot

LONG_HEADER = ["event","timestamp_ticks","trigger_id","channel","sample","adc"]
WIDE_PREFIX = ["event_index","timestamp_ticks","timestamp_ns","trigger_id",
               "channel","n_samples","sample_period_ns"]

def open_text(path: Path, mode="r"):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode + "t", newline="")
    return path.open(mode, newline="")

def detect_layout(path: Path):
    with open_text(path, "r") as f:
        row = next(csv.reader(f), None)
    if row == LONG_HEADER:
        return "long"
    if row and row[:len(WIDE_PREFIX)] == WIDE_PREFIX:
        return "wide"
    raise RuntimeError(f"Unrecognized ALPINE CSV header: {row}")

def inspect_csv(path: Path):
    layout = detect_layout(path)
    with open_text(path, "r") as f:
        r = csv.DictReader(f)
        first = next(r, None)
        cols = r.fieldnames
    return {"layout": layout, "columns": cols, "first_row": first}

def _infer_long(path: Path):
    with open_text(path, "r") as f:
        r = csv.DictReader(f)
        if r.fieldnames != LONG_HEADER:
            raise RuntimeError(f"Unexpected header: {r.fieldnames}")
        first_event = None
        channel_samples = {}
        channel_order = []
        for row in r:
            ev = int(row["event"])
            if first_event is None:
                first_event = ev
            if ev != first_event:
                break
            ch = int(row["channel"])
            if ch not in channel_samples:
                channel_order.append(ch)
                channel_samples[ch] = []
            channel_samples[ch].append(int(row["sample"]))
    if first_event is None:
        raise RuntimeError("CSV has no data")
    n = len(channel_samples[channel_order[0]])
    expected = list(range(n))
    for ch in channel_order:
        if channel_samples[ch] != expected:
            raise RuntimeError(f"First event ch{ch:02d} is not samples 0..{n-1}")
    return channel_order, n

def _parse_numeric_block(block: bytes):
    block = block.strip(b"\r\n")
    if not block:
        return np.empty((0,6), dtype=np.int64)
    vals = np.fromstring(block.replace(b"\r", b"").replace(b"\n", b","), sep=",", dtype=np.int64)
    if vals.size % 6:
        raise RuntimeError(f"Parsed {vals.size} fields; expected a multiple of 6")
    return vals.reshape(-1, 6)

def long_to_root(source: Path, output: Path, overwrite=False, block_mb=64):
    if output.exists() and not overwrite:
        return {"status":"skipped","reason":"output exists"}

    channels, n_samples = _infer_long(source)
    rows_per_event = len(channels) * n_samples
    output.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(output) + ".part")
    part.unlink(missing_ok=True)

    opener = gzip.open if source.name.endswith(".gz") else open
    carry_text = b""
    carry_rows = np.empty((0,6), dtype=np.int64)
    total_events = 0
    tree = None

    with opener(source, "rb") as f, uproot.recreate(part) as rf:
        header = f.readline().decode("ascii").strip().split(",")
        if header != LONG_HEADER:
            raise RuntimeError(f"Unexpected header: {header}")

        while True:
            chunk = f.read(block_mb * 1024 * 1024)
            eof = not chunk
            buf = carry_text + chunk
            if eof:
                complete = buf
                carry_text = b""
            else:
                pos = buf.rfind(b"\n")
                if pos < 0:
                    carry_text = buf
                    continue
                complete = buf[:pos]
                carry_text = buf[pos+1:]

            rows = _parse_numeric_block(complete)
            if len(carry_rows):
                rows = np.concatenate((carry_rows, rows), axis=0) if len(rows) else carry_rows
                carry_rows = np.empty((0,6), dtype=np.int64)

            n_events = len(rows) // rows_per_event
            n_full = n_events * rows_per_event
            if n_events:
                full = rows[:n_full].reshape(n_events, len(channels), n_samples, 6)
                carry_rows = rows[n_full:].copy()

                exp_ch = np.asarray(channels, dtype=np.int64)
                if not np.array_equal(full[:,:,0,3], np.broadcast_to(exp_ch, (n_events,len(channels)))):
                    raise RuntimeError("Channel order changed or rows are not event-major")
                if np.any(full[:,:,0,4] != 0) or np.any(full[:,:,-1,4] != n_samples-1):
                    raise RuntimeError("Sample index boundaries are inconsistent")

                ev = full[:,0,0,0]
                ticks = full[:,0,0,1]
                trig = full[:,0,0,2]
                if np.any(full[:,-1,-1,0] != ev):
                    raise RuntimeError("Event changes within waveform block")
                if np.any(full[:,-1,-1,1] != ticks):
                    raise RuntimeError("Timestamp changes within event")
                if np.any(full[:,-1,-1,2] != trig):
                    raise RuntimeError("Trigger ID changes within event")

                adc = full[...,5]
                if np.any(adc < 0) or np.any(adc > 65535):
                    raise RuntimeError("ADC outside uint16 range")

                arrays = {
                    "event": ev.astype(np.uint64, copy=False),
                    "timestamp_ticks": ticks.astype(np.uint64, copy=False),
                    "trigger_id": trig.astype(np.uint32, copy=False),
                }
                adc = adc.astype(np.uint16, copy=False)
                for i, ch in enumerate(channels):
                    arrays[f"ch{ch:02d}"] = adc[:,i,:]

                if tree is None:
                    tree = rf.mktree("Events", arrays)
                else:
                    tree.extend(arrays)
                total_events += n_events

            if eof:
                break

    if len(carry_rows):
        raise RuntimeError(f"CSV ended with {len(carry_rows)} incomplete rows")
    if tree is None:
        raise RuntimeError("No events written")

    os.replace(part, output)
    try: output.chmod(0o666)
    except OSError: pass
    return {"status":"ok","events":total_events,"channels":channels,"samples_per_channel":n_samples}

def wide_to_root(source: Path, output: Path, overwrite=False, chunk_events=2000):
    if output.exists() and not overwrite:
        return {"status":"skipped","reason":"output exists"}

    with open_text(source, "r") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        if cols[:len(WIDE_PREFIX)] != WIDE_PREFIX:
            raise RuntimeError("Unexpected wide CSV header")
        sample_cols = [c for c in cols if c.startswith("sample_")]
        n_samples = len(sample_cols)
        first_event = None
        channels = []
        for row in r:
            ev = int(row["event_index"])
            if first_event is None: first_event = ev
            if ev != first_event: break
            channels.append(int(row["channel"]))
    if not channels or not n_samples:
        raise RuntimeError("Could not infer wide CSV schema")

    output.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(output) + ".part")
    part.unlink(missing_ok=True)
    tree = None
    total = 0

    def emit(rf, events):
        nonlocal tree, total
        if not events: return
        n = len(events)
        arrays = {
            "event": np.empty(n, dtype=np.uint64),
            "timestamp_ticks": np.empty(n, dtype=np.uint64),
            "trigger_id": np.empty(n, dtype=np.uint32),
        }
        for ch in channels:
            arrays[f"ch{ch:02d}"] = np.empty((n,n_samples), dtype=np.uint16)
        for i, rows in enumerate(events):
            arrays["event"][i] = int(rows[0]["event_index"])
            arrays["timestamp_ticks"][i] = int(rows[0]["timestamp_ticks"])
            arrays["trigger_id"][i] = int(rows[0]["trigger_id"])
            seen = set()
            for row in rows:
                ch = int(row["channel"])
                seen.add(ch)
                arrays[f"ch{ch:02d}"][i,:] = [int(row[c]) for c in sample_cols]
            if seen != set(channels):
                raise RuntimeError(f"Channel set changed in event {arrays['event'][i]}")
        if tree is None:
            tree = rf.mktree("Events", arrays)
        else:
            tree.extend(arrays)
        total += n

    with uproot.recreate(part) as rf, open_text(source, "r") as f:
        r = csv.DictReader(f)
        current = None
        rows = []
        chunk = []
        for row in r:
            ev = int(row["event_index"])
            if current is None: current = ev
            if ev != current:
                chunk.append(rows)
                rows = []
                current = ev
                if len(chunk) >= chunk_events:
                    emit(rf, chunk)
                    chunk = []
            rows.append(row)
        if rows: chunk.append(rows)
        emit(rf, chunk)

    os.replace(part, output)
    try: output.chmod(0o666)
    except OSError: pass
    return {"status":"ok","events":total,"channels":channels,"samples_per_channel":n_samples}

def csv_to_root(source: Path, output: Path, overwrite=False):
    layout = detect_layout(source)
    result = long_to_root(source, output, overwrite) if layout == "long" else wide_to_root(source, output, overwrite)
    result["layout"] = layout
    return result
