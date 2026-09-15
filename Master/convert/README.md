# alpine-convert 1.0.0

Reliable bidirectional conversion for ALPINE waveform files on the UZH farm.

Supported directions:

- `.raw` -> `.csv.gz`
- `.root` -> `.csv.gz`
- original ALPINE `.csv.gz` -> `.root`
- package-generated wide `.csv(.gz)` -> `.root`
- terminal/local conversion
- Slurm array conversion using worker-local `/scratch`

Inputs are never deleted.

## Install

Keep the package source on the large GFS filesystem, but keep the Python virtualenv
in your home directory. This avoids filesystem permission/metadata problems with
virtualenv files on GFS.

### Easiest installation

```bash
cd /disk/gfs_atp/xlzd/alpine/convert/alpine-convert-1.0.0
./install.sh
```

By default this creates/updates:

```text
$HOME/.venvs/alpine-convert
```

Activate it later with:

```bash
source ~/.venvs/alpine-convert/bin/activate
```

### Reuse the existing working ALPINE virtualenv

The environment used successfully during development can instead be updated directly:

```bash
source /home/atp/xlzd/ALPINE/convert/.venv/bin/activate
cd /disk/gfs_atp/xlzd/alpine/convert/alpine-convert-1.0.0
python -m pip install --upgrade .
```

Verify:

```bash
alpine-convert --version
```

Expected:

```text
1.0.0
```

## Inspect

```bash
alpine-convert inspect waveforms-0000.root
alpine-convert inspect waveforms-0000.raw
alpine-convert inspect waveforms-0000.csv.gz
```

## Local conversion

ROOT -> compressed CSV:

```bash
alpine-convert convert waveforms-0000.root
```

RAW -> compressed CSV:

```bash
alpine-convert convert waveforms-0000.raw
```

Original ALPINE CSV.GZ -> ROOT:

```bash
alpine-convert convert waveforms-0000.csv.gz
```

Generated wide CSV.GZ -> ROOT:

```bash
alpine-convert convert waveforms-0000.root.csv.gz --to root
```

Outputs are skipped if they already exist. Use `--overwrite` deliberately if replacement is intended.

## CSV layouts

Default export is **wide**:

```text
event_index,timestamp_ticks,timestamp_ns,trigger_id,channel,n_samples,
sample_period_ns,sample_0000,...,sample_0999
```

This is reversible and much more compact than the original long representation.

To recreate the original six-column ALPINE layout:

```bash
alpine-convert convert waveforms-0000.root --csv-layout long
```

Long layout:

```text
event,timestamp_ticks,trigger_id,channel,sample,adc
```

To create plain `.csv` instead of `.csv.gz`:

```bash
alpine-convert convert waveforms-0000.root --no-gzip
```

## Directory dry-run

A directory requires an explicit direction:

```bash
alpine-convert convert /disk/gfs_atp/xlzd/alpine/run1_2 \
  --to csv --dry-run
```

or:

```bash
alpine-convert convert /disk/gfs_atp/xlzd/alpine/run1_2 \
  --to root --dry-run
```

## Slurm: ROOT/RAW -> CSV.GZ

```bash
alpine-convert submit /disk/gfs_atp/xlzd/alpine/run1_2 \
  --to csv \
  --partition express \
  --max-parallel 4 \
  --mem 4G \
  --time 02:00:00
```

## Slurm: CSV/CSV.GZ -> ROOT

```bash
alpine-convert submit /disk/gfs_atp/xlzd/alpine/run1_2 \
  --to root \
  --partition express \
  --max-parallel 4 \
  --mem 4G \
  --time 02:00:00
```

The Slurm workflow:

1. chooses only missing outputs;
2. submits one array task per file;
3. copies input to `/scratch/$USER/alpine-convert/...`;
4. converts on worker-local storage;
5. copies the completed file back atomically;
6. deletes scratch only after success;
7. retains scratch on failure for debugging.

The package deliberately preserves the virtualenv Python path when submitting jobs. It does **not** resolve `.venv/bin/python` to `/usr/bin/python`, which was the cause of the earlier farm failure.

## Test a small number first

```bash
alpine-convert submit /disk/gfs_atp/xlzd/alpine/run1_2 \
  --to root \
  --limit 2 \
  --max-parallel 2
```

## Monitor jobs

The command prints the job ID:

```bash
squeue -j JOBID
```

After completion:

```bash
sacct -j JOBID \
  --format=JobID,State,ExitCode,Elapsed,NodeList
```

Logs are under:

```text
<source>/.alpine_convert/logs/
```

unless `--state-dir` is specified.

## Naming

```text
waveforms-0000.root
 -> waveforms-0000.root.csv.gz

waveforms-0000.raw
 -> waveforms-0000.raw.csv.gz

waveforms-0000.csv.gz
 -> waveforms-0000.root

waveforms-0000.root.csv.gz
 -> waveforms-0000.root

waveforms-0000.raw.csv.gz
 -> waveforms-0000.raw.root
```

The extension-preserving CSV names prevent collisions when RAW and ROOT with the same basename coexist.

## Recommended use

For multi-GB conversions, use `submit`; do not perform bulk conversion on `farm-ui1`.

For quick inspection, one-file tests, or small files, use `convert`.

For the current shared repository:

```bash
cd /disk/gfs_atp/xlzd/alpine
source /home/atp/xlzd/ALPINE/convert/.venv/bin/activate
```

then run `alpine-convert`.
