from __future__ import annotations
import re
from pathlib import Path
import uproot

class RootFormatError(RuntimeError):
    pass

def _tree_names(f):
    names = []
    for name, classname in f.classnames(recursive=True).items():
        if "TTree" in classname or "RNTuple" in classname:
            names.append(name.split(";")[0])
    return list(dict.fromkeys(names))

def select_tree(f, requested=None):
    names = _tree_names(f)
    if requested:
        if requested not in names:
            raise RootFormatError(f"Tree {requested!r} not found; available={names}")
        return requested, f[requested]
    if "Events" in names:
        return "Events", f["Events"]
    if len(names) == 1:
        return names[0], f[names[0]]
    if not names:
        raise RootFormatError("No TTree/RNTuple found")
    raise RootFormatError(f"Multiple trees found: {names}; pass --tree")

def detect_schema(tree):
    keys = list(tree.keys())
    if not {"event", "timestamp_ticks", "trigger_id"}.issubset(keys):
        return None
    chans = []
    n_samples = None
    for key in keys:
        m = re.fullmatch(r"ch(\d+)", key)
        if not m:
            continue
        typename = getattr(tree[key], "typename", "")
        lm = re.search(r"\[(\d+)\]", typename)
        if not lm:
            return None
        n = int(lm.group(1))
        if n_samples is None:
            n_samples = n
        elif n != n_samples:
            raise RootFormatError("Channel branches have different waveform lengths")
        chans.append((int(m.group(1)), key))
    chans.sort()
    return (chans, n_samples) if chans and n_samples else None

def inspect_root(path: Path, requested=None):
    with uproot.open(path) as f:
        name, tree = select_tree(f, requested)
        schema = detect_schema(tree)
        return {
            "tree": name,
            "entries": tree.num_entries,
            "branches": [{"name": k, "typename": getattr(tree[k], "typename", "")}
                         for k in tree.keys()],
            "alpine_schema": None if schema is None else {
                "channels": [c for c, _ in schema[0]],
                "samples_per_channel": schema[1],
            },
        }
