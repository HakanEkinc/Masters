from __future__ import annotations
import csv, gzip, os
from pathlib import Path
import numpy as np
import uproot
from .raw import iter_events
from .rootio import select_tree, detect_schema, RootFormatError
from .csvroot import LONG_HEADER, WIDE_PREFIX

def _writer(path: Path):
    if path.name.endswith(".gz") or path.name.endswith(".gz.part"):
        f = gzip.open(path, "wt", newline="", compresslevel=4)
    else:
        f = path.open("w", newline="")
    return f, csv.writer(f)

def _wide_header(n):
    return WIDE_PREFIX + [f"sample_{i:04d}" for i in range(n)]

def raw_to_csv(source: Path, output: Path, layout="wide", overwrite=False):
    if output.exists() and not overwrite:
        return {"status":"skipped","reason":"output exists"}
    output.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(output)+".part")
    part.unlink(missing_ok=True)
    events = rows = 0
    fh, w = _writer(part)
    try:
        meta0 = None
        for meta, ev, ticks, trig, channels in iter_events(source):
            if meta0 is None:
                meta0 = meta
                n_samples = int(meta.get("recordLengthSamples", len(channels[0][1])))
                w.writerow(_wide_header(n_samples) if layout=="wide" else LONG_HEADER)
            sp = meta.get("samplePeriodNs","")
            tns = ticks*sp if isinstance(sp,(int,float)) else ""
            for ch, samples in channels:
                if layout == "wide":
                    w.writerow([ev,ticks,tns,trig,ch,len(samples),sp,*samples]); rows += 1
                else:
                    for i, adc in enumerate(samples):
                        w.writerow([ev,ticks,trig,ch,i,adc]); rows += 1
            events += 1
    finally:
        fh.close()
    os.replace(part, output)
    try: output.chmod(0o666)
    except OSError: pass
    return {"status":"ok","events":events,"rows":rows}

def root_to_csv(source: Path, output: Path, layout="wide", overwrite=False, tree_name=None, step_size="100 MB"):
    if output.exists() and not overwrite:
        return {"status":"skipped","reason":"output exists"}
    output.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(output)+".part")
    part.unlink(missing_ok=True)
    events = rows = 0

    with uproot.open(source) as f:
        name, tree = select_tree(f, tree_name)
        schema = detect_schema(tree)
        if schema is None:
            raise RootFormatError("Not recognized ALPINE waveform ROOT schema")
        channels, n_samples = schema
        expr = ["event","timestamp_ticks","trigger_id"] + [b for _,b in channels]
        fh, w = _writer(part)
        try:
            w.writerow(_wide_header(n_samples) if layout=="wide" else LONG_HEADER)
            for arr in tree.iterate(expressions=expr, step_size=step_size, library="np"):
                for i in range(len(arr["event"])):
                    ev = int(arr["event"][i]); ticks = int(arr["timestamp_ticks"][i]); trig = int(arr["trigger_id"][i])
                    for ch, branch in channels:
                        s = np.asarray(arr[branch][i], dtype=np.uint16)
                        if layout=="wide":
                            w.writerow([ev,ticks,"",trig,ch,len(s),"",*s.tolist()]); rows += 1
                        else:
                            for j, adc in enumerate(s):
                                w.writerow([ev,ticks,trig,ch,j,int(adc)]); rows += 1
                    events += 1
        finally:
            fh.close()
    os.replace(part, output)
    try: output.chmod(0o666)
    except OSError: pass
    return {"status":"ok","tree":name,"events":events,"rows":rows}
