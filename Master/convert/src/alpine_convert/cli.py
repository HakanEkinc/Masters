from __future__ import annotations
import argparse, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path
from . import __version__
from .raw import inspect_raw
from .rootio import inspect_root
from .csvroot import inspect_csv, csv_to_root
from .export import root_to_csv, raw_to_csv

def is_csv(p): return p.name.endswith(".csv") or p.name.endswith(".csv.gz")

def direction(p):
    if p.suffix.lower() in {".root",".raw"}: return "csv"
    if is_csv(p): return "root"
    raise ValueError(f"Unsupported input: {p}")

def output_for(src: Path, to: str, gzip_output=True):
    if to=="csv":
        return src.with_name(src.name + (".csv.gz" if gzip_output else ".csv"))
    name = src.name
    if name.endswith(".csv.gz"): name = name[:-7]
    elif name.endswith(".csv"): name = name[:-4]
    else: raise ValueError(f"Not CSV: {src}")
    if name.endswith(".root"): return src.with_name(name)
    if name.endswith(".raw"): return src.with_name(name + ".root")
    return src.with_name(name + ".root")

def discover(base: Path, to: str):
    if base.is_file(): return [base]
    if to=="csv":
        return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".root",".raw"})
    candidates = sorted(p for p in base.rglob("*") if p.is_file() and is_csv(p))
    # Deduplicate multiple CSV representations targeting the same ROOT.
    chosen = {}
    def score(p):
        n=p.name
        return (1 if ".root.csv" in n or ".raw.csv" in n else 0, len(n), str(p))
    for p in sorted(candidates, key=score):
        chosen.setdefault(str(output_for(p,"root")), p)
    return sorted(chosen.values())

def mkdir_shared(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    try: p.chmod(0o2777)
    except OSError:
        try: p.chmod(0o777)
        except OSError: pass

def chmod_shared(p: Path):
    try: p.chmod(0o666)
    except OSError: pass

def convert_one(src, dst, to, layout, overwrite, tree, step_size):
    print(f"Input : {src}", flush=True); print(f"Output: {dst}", flush=True)
    t=time.time()
    if to=="root":
        result=csv_to_root(src,dst,overwrite)
    elif src.suffix.lower()==".raw":
        result=raw_to_csv(src,dst,layout,overwrite)
    else:
        result=root_to_csv(src,dst,layout,overwrite,tree,step_size)
    if result["status"]=="ok": chmod_shared(dst)
    print(f"[{result['status'].upper()}] {result} ({time.time()-t:.1f}s)", flush=True)
    return result

def cmd_inspect(a):
    p=Path(a.path).expanduser().resolve()
    if p.suffix.lower()==".raw": obj=inspect_raw(p,a.events)
    elif p.suffix.lower()==".root": obj=inspect_root(p,a.tree)
    elif is_csv(p): obj=inspect_csv(p)
    else: raise SystemExit(f"Unsupported: {p}")
    print(json.dumps(obj,indent=2,default=str))

def resolve_to(base,to):
    if to!="auto": return to
    if base.is_dir(): raise SystemExit("Directory conversion requires --to csv or --to root")
    return direction(base)

def cmd_convert(a):
    base=Path(a.path).expanduser().resolve(); to=resolve_to(base,a.to)
    files=discover(base,to)
    jobs=[(s, output_for(s,to,not a.no_gzip)) for s in files]
    if a.output:
        if len(jobs)!=1: raise SystemExit("--output only valid for one input file")
        jobs=[(jobs[0][0],Path(a.output).expanduser().resolve())]
    pending=[j for j in jobs if a.overwrite or not j[1].exists()]
    print(f"Direction: {to}\nFound: {len(files)}\nAlready converted: {len(jobs)-len(pending)}\nTo convert: {len(pending)}")
    if a.dry_run:
        for s,d in pending: print(f"{s} -> {d}")
        return
    failed=0
    for s,d in pending:
        try: convert_one(s,d,to,a.csv_layout,a.overwrite,a.tree,a.step_size)
        except Exception as e:
            failed+=1; print(f"[ERROR] {type(e).__name__}: {e}",file=sys.stderr)
    raise SystemExit(1 if failed else 0)

def cmd_submit(a):
    base=Path(a.path).expanduser().resolve(); to=resolve_to(base,a.to)
    files=discover(base,to)
    jobs=[]
    for s in files:
        d=output_for(s,to,not a.no_gzip)
        if a.overwrite or not d.exists():
            jobs.append((s,d))
    if a.limit is not None: jobs=jobs[:a.limit]
    print(f"Direction: {to}\nFound: {len(files)}\nTo submit: {len(jobs)}\nPartition: {a.partition}\nMax simultaneous: {a.max_parallel}")
    if not jobs: print("Nothing to submit."); return
    state=Path(a.state_dir).expanduser().resolve() if a.state_dir else ((base if base.is_dir() else base.parent)/".alpine_convert")
    stamp=time.strftime("%Y%m%dT%H%M%S")
    manifest=state/"manifests"/f"{to}-{stamp}.jsonl"; logs=state/"logs"
    mkdir_shared(manifest.parent); mkdir_shared(logs)
    with manifest.open("w") as f:
        for s,d in jobs:
            f.write(json.dumps({"input":str(s.resolve()),"output":str(d.resolve()),"to":to,"layout":a.csv_layout})+"\n")
    chmod_shared(manifest)

    # Preserve venv path exactly; DO NOT resolve symlinks to /usr/bin/python.
    py=Path(sys.executable)
    worker=[str(py),"-m","alpine_convert.cli","_worker","--manifest",str(manifest),
            "--scratch-root",a.scratch_root,"--step-size",a.step_size]
    if a.overwrite: worker.append("--overwrite")
    if a.tree: worker += ["--tree",a.tree]
    sb=["sbatch","--parsable","--partition",a.partition,"--job-name",f"alpine-{to}",
        "--array",f"0-{len(jobs)-1}%{a.max_parallel}","--cpus-per-task",str(a.cpus_per_task),
        "--mem",a.mem,"--time",a.time,
        "--output",str(logs/"slurm-%A_%a.out"),"--error",str(logs/"slurm-%A_%a.err")]
    if a.account: sb += ["--account",a.account]
    sb += ["--wrap","umask 000; "+" ".join(shlex.quote(x) for x in worker)]
    r=subprocess.run(sb,check=True,text=True,capture_output=True)
    jid=r.stdout.strip()
    print(f"Submitted Slurm job array: {jid}\nCheck: squeue -j {jid}\nLogs: {logs}")

def copy_back(src: Path,dst: Path):
    mkdir_shared(dst.parent)
    jid=os.environ.get("SLURM_ARRAY_JOB_ID",os.environ.get("SLURM_JOB_ID","job"))
    tid=os.environ.get("SLURM_ARRAY_TASK_ID","0")
    tmp=dst.with_name(dst.name+f".copying.{jid}_{tid}")
    tmp.unlink(missing_ok=True)
    shutil.copyfile(src,tmp); chmod_shared(tmp); os.replace(tmp,dst); chmod_shared(dst)

def cmd_worker(a):
    tid=int(os.environ["SLURM_ARRAY_TASK_ID"]); jid=os.environ["SLURM_JOB_ID"]
    with Path(a.manifest).open() as f: jobs=[json.loads(x) for x in f if x.strip()]
    j=jobs[tid]
    src=Path(j["input"]); dst=Path(j["output"]); to=j["to"]; layout=j["layout"]
    if dst.exists() and not a.overwrite:
        print(f"[SKIP] {dst} exists"); return
    scratch=Path(a.scratch_root)/os.environ.get("USER","unknown")/"alpine-convert"/f"{jid}_{tid}"
    scratch.mkdir(parents=True,exist_ok=False)
    local_in=scratch/src.name; local_out=scratch/dst.name
    ok=False
    try:
        print(f"Worker: {os.uname().nodename}\nSource: {src}\nScratch: {scratch}\nDestination: {dst}",flush=True)
        shutil.copyfile(src,local_in)
        convert_one(local_in,local_out,to,layout,True,a.tree,a.step_size)
        copy_back(local_out,dst); ok=True; print(f"[DONE] {dst}",flush=True)
    finally:
        if ok: shutil.rmtree(scratch,ignore_errors=True)
        else: print(f"[NOTE] failure; scratch retained at {scratch}",file=sys.stderr)

def add_common(x):
    x.add_argument("path")
    x.add_argument("--to",choices=["auto","csv","root"],default="auto")
    x.add_argument("--csv-layout",choices=["wide","long"],default="wide")
    x.add_argument("--no-gzip",action="store_true")
    x.add_argument("--overwrite",action="store_true")
    x.add_argument("--tree")
    x.add_argument("--step-size",default="100 MB")

def parser():
    p=argparse.ArgumentParser(prog="alpine-convert",description="Bidirectional ALPINE RAW/ROOT/CSV converter")
    p.add_argument("--version",action="version",version=__version__)
    sub=p.add_subparsers(dest="cmd",required=True)
    x=sub.add_parser("inspect"); x.add_argument("path"); x.add_argument("--events",type=int,default=3); x.add_argument("--tree"); x.set_defaults(func=cmd_inspect)
    x=sub.add_parser("convert",help="convert locally"); add_common(x); x.add_argument("--output"); x.add_argument("--dry-run",action="store_true"); x.set_defaults(func=cmd_convert)
    x=sub.add_parser("submit",help="submit Slurm array"); add_common(x)
    x.add_argument("--partition",default="express"); x.add_argument("--account")
    x.add_argument("--max-parallel",type=int,default=4); x.add_argument("--cpus-per-task",type=int,default=1)
    x.add_argument("--mem",default="4G"); x.add_argument("--time",default="02:00:00")
    x.add_argument("--scratch-root",default="/scratch"); x.add_argument("--state-dir"); x.add_argument("--limit",type=int)
    x.set_defaults(func=cmd_submit)
    x=sub.add_parser("_worker"); x.add_argument("--manifest",required=True); x.add_argument("--scratch-root",default="/scratch")
    x.add_argument("--overwrite",action="store_true"); x.add_argument("--tree"); x.add_argument("--step-size",default="100 MB"); x.set_defaults(func=cmd_worker)
    return p

def main():
    a=parser().parse_args(); a.func(a)

if __name__=="__main__":
    main()
