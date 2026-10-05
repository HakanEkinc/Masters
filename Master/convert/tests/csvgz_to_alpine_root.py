#!/usr/bin/env python3
"""
Reconstruct an ALPINE waveform ROOT TTree from the long-form CSV.GZ format:

    event,timestamp_ticks,trigger_id,channel,sample,adc

Output TTree "Events":
    event             uint64
    timestamp_ticks   uint64
    trigger_id        uint32
    chNN              uint16[Nsamples]

The converter is streaming/chunked and does not load the full CSV into memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np
import uproot

EXPECTED_HEADER = [
    "event",
    "timestamp_ticks",
    "trigger_id",
    "channel",
    "sample",
    "adc",
]


def inspect_first_event(path: Path):
    """Infer ordered channels and samples/channel from only the first event."""
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("CSV.GZ is empty")

        if header != EXPECTED_HEADER:
            raise RuntimeError(
                f"Unexpected header {header!r}; expected {EXPECTED_HEADER!r}"
            )

        first_event = None
        by_channel: dict[int, list[int]] = {}

        for row in reader:
            if len(row) != 6:
                raise RuntimeError(f"Malformed CSV row: {row!r}")

            event = int(row[0])

            if first_event is None:
                first_event = event

            if event != first_event:
                break

            channel = int(row[3])
            sample = int(row[4])
            by_channel.setdefault(channel, []).append(sample)

    if first_event is None:
        raise RuntimeError("CSV contains a header but no data rows")

    channels = list(by_channel)
    if not channels:
        raise RuntimeError("Could not infer channels")

    n_samples = len(by_channel[channels[0]])
    expected_samples = list(range(n_samples))

    for channel in channels:
        samples = by_channel[channel]
        if samples != expected_samples:
            raise RuntimeError(
                f"First event channel {channel} does not contain ordered "
                f"sample indices 0..{n_samples - 1}"
            )

    return first_event, channels, n_samples


def parse_numeric_block(block: bytes) -> np.ndarray:
    """Parse complete CSV data lines into an (N, 6) int64 array."""
    block = block.strip(b"\r\n")
    if not block:
        return np.empty((0, 6), dtype=np.int64)

    # np.fromstring performs the numeric parsing in compiled NumPy code and is
    # much faster than a Python csv loop over hundreds of millions of rows.
    numeric = block.replace(b"\r", b"").replace(b"\n", b",")
    values = np.fromstring(numeric, sep=",", dtype=np.int64)

    if values.size % 6 != 0:
        raise RuntimeError(
            f"Parsed {values.size} numeric fields, which is not divisible by 6"
        )

    return values.reshape(-1, 6)


def make_root_arrays(
    rows: np.ndarray,
    channels: list[int],
    n_samples: int,
):
    n_channels = len(channels)
    rows_per_event = n_channels * n_samples

    if len(rows) % rows_per_event:
        raise RuntimeError("Internal error: incomplete event passed to writer")

    n_events = len(rows) // rows_per_event
    shaped = rows.reshape(n_events, n_channels, n_samples, 6)

    expected_channels = np.asarray(channels, dtype=np.int64)

    # Validate the structural boundaries for every event without scanning every
    # field repeatedly.
    if not np.array_equal(
        shaped[:, :, 0, 3],
        np.broadcast_to(expected_channels, (n_events, n_channels)),
    ):
        raise RuntimeError("Channel order changed or CSV rows are not event-major")

    if not np.array_equal(
        shaped[:, :, -1, 3],
        np.broadcast_to(expected_channels, (n_events, n_channels)),
    ):
        raise RuntimeError("Channel changed within a waveform")

    if np.any(shaped[:, :, 0, 4] != 0):
        raise RuntimeError("Waveform does not start at sample 0")

    if np.any(shaped[:, :, -1, 4] != n_samples - 1):
        raise RuntimeError(
            f"Waveform does not end at sample {n_samples - 1}"
        )

    event = shaped[:, 0, 0, 0]
    timestamp = shaped[:, 0, 0, 1]
    trigger = shaped[:, 0, 0, 2]

    if np.any(shaped[:, -1, -1, 0] != event):
        raise RuntimeError("Event number changes within an event")
    if np.any(shaped[:, -1, -1, 1] != timestamp):
        raise RuntimeError("timestamp_ticks changes within an event")
    if np.any(shaped[:, -1, -1, 2] != trigger):
        raise RuntimeError("trigger_id changes within an event")

    adc = shaped[..., 5]
    if np.any(adc < 0) or np.any(adc > 65535):
        raise RuntimeError("ADC value outside uint16 range")

    arrays = {
        "event": event.astype(np.uint64, copy=False),
        "timestamp_ticks": timestamp.astype(np.uint64, copy=False),
        "trigger_id": trigger.astype(np.uint32, copy=False),
    }

    adc = adc.astype(np.uint16, copy=False)

    for index, channel in enumerate(channels):
        arrays[f"ch{channel:02d}"] = adc[:, index, :]

    return arrays


def convert(
    source: Path,
    output: Path,
    block_mb: int = 64,
    overwrite: bool = False,
):
    source = source.resolve()
    output = output.resolve()

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    first_event, channels, n_samples = inspect_first_event(source)

    print(f"Input             : {source}", flush=True)
    print(f"Output            : {output}", flush=True)
    print(f"First event       : {first_event}", flush=True)
    print(f"Channels          : {channels}", flush=True)
    print(f"Samples/channel   : {n_samples}", flush=True)

    rows_per_event = len(channels) * n_samples
    block_bytes = block_mb * 1024 * 1024

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        temporary.unlink()

    text_carry = b""
    row_carry = np.empty((0, 6), dtype=np.int64)
    total_events = 0
    started = time.time()

    try:
        with gzip.open(source, "rb") as gz, uproot.recreate(temporary) as root:
            header = gz.readline().decode("ascii").strip().split(",")
            if header != EXPECTED_HEADER:
                raise RuntimeError(
                    f"Unexpected header {header!r}; expected {EXPECTED_HEADER!r}"
                )

            tree = None

            while True:
                chunk = gz.read(block_bytes)
                at_eof = not chunk
                buffer = text_carry + chunk

                if at_eof:
                    complete = buffer
                    text_carry = b""
                else:
                    last_newline = buffer.rfind(b"\n")
                    if last_newline < 0:
                        text_carry = buffer
                        continue
                    complete = buffer[:last_newline]
                    text_carry = buffer[last_newline + 1 :]

                rows = parse_numeric_block(complete)

                if len(row_carry):
                    if len(rows):
                        rows = np.concatenate((row_carry, rows), axis=0)
                    else:
                        rows = row_carry
                    row_carry = np.empty((0, 6), dtype=np.int64)

                complete_event_count = len(rows) // rows_per_event
                complete_row_count = complete_event_count * rows_per_event

                if complete_event_count:
                    event_rows = rows[:complete_row_count]
                    row_carry = rows[complete_row_count:].copy()

                    arrays = make_root_arrays(
                        event_rows,
                        channels=channels,
                        n_samples=n_samples,
                    )

                    if tree is None:
                        # Passing actual arrays to mktree lets uproot infer the
                        # fixed-length chNN[N] branch types and immediately
                        # writes the first chunk.
                        tree = root.mktree("Events", arrays)
                    else:
                        tree.extend(arrays)

                    total_events += complete_event_count
                    elapsed = time.time() - started
                    print(
                        f"\rEvents written: {total_events:,} "
                        f"({elapsed:.1f} s)",
                        end="",
                        flush=True,
                    )
                else:
                    row_carry = rows.copy()

                if at_eof:
                    break

            print(flush=True)

            if len(row_carry):
                raise RuntimeError(
                    f"CSV ended with {len(row_carry)} rows, not a complete "
                    f"{rows_per_event}-row event"
                )

            if tree is None:
                raise RuntimeError("No complete events were written")

        os.replace(temporary, output)
        try:
            output.chmod(0o666)
        except OSError:
            pass

    except Exception:
        # Keep no apparently-valid final ROOT on failure. The .part file is
        # intentionally retained for debugging.
        raise

    elapsed = time.time() - started
    print(
        f"[OK] Wrote {total_events:,} events to {output} in {elapsed:.1f} s",
        flush=True,
    )



def compare_root_files(reference: Path, candidate: Path):
    """Compare structure and representative entry windows."""
    print(f"Comparing against : {reference}", flush=True)

    with uproot.open(reference) as ref_file, uproot.open(candidate) as out_file:
        ref = ref_file["Events"]
        out = out_file["Events"]

        ref_branches = list(ref.keys())
        out_branches = list(out.keys())

        if ref_branches != out_branches:
            raise RuntimeError(
                f"Branch mismatch:\nreference={ref_branches}\ncandidate={out_branches}"
            )

        if ref.num_entries != out.num_entries:
            raise RuntimeError(
                f"Entry mismatch: reference={ref.num_entries}, "
                f"candidate={out.num_entries}"
            )

        for name in ref_branches:
            if ref[name].typename != out[name].typename:
                raise RuntimeError(
                    f"Type mismatch for {name}: "
                    f"{ref[name].typename!r} != {out[name].typename!r}"
                )

        n = ref.num_entries
        windows = [(0, min(5, n))]
        if n > 10:
            mid = n // 2
            windows.append((max(0, mid - 2), min(n, mid + 3)))
        if n > 5:
            windows.append((max(0, n - 5), n))

        for start, stop in windows:
            ref_arrays = ref.arrays(
                ref_branches,
                entry_start=start,
                entry_stop=stop,
                library="np",
            )
            out_arrays = out.arrays(
                out_branches,
                entry_start=start,
                entry_stop=stop,
                library="np",
            )

            for name in ref_branches:
                if not np.array_equal(ref_arrays[name], out_arrays[name]):
                    raise RuntimeError(
                        f"Data mismatch in branch {name}, entries {start}:{stop}"
                    )

    print(
        f"[COMPARE OK] Same branches, types, {ref.num_entries:,} entries, "
        "and sampled waveform data.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ALPINE long-form CSV.GZ waveforms back to ROOT."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Default: replace .csv.gz with .root",
    )
    parser.add_argument("--block-mb", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compare-reference",
        type=Path,
        help="After conversion, compare structure and sampled entries to this ROOT file.",
    )
    args = parser.parse_args()

    source = args.input

    if args.output is None:
        if not source.name.endswith(".csv.gz"):
            parser.error("Input must end in .csv.gz when output is omitted")
        output = source.with_name(source.name[:-7] + ".root")
    else:
        output = args.output

    try:
        convert(
            source,
            output,
            block_mb=args.block_mb,
            overwrite=args.overwrite,
        )
        if args.compare_reference is not None:
            compare_root_files(
                args.compare_reference.resolve(),
                output.resolve(),
            )
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

