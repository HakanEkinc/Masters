# ALPINE processing, version 2

Replace only `Master/Python/process` with this folder. `../func_notebook.py` and
all notebooks stay unchanged. The script finds the helper relative to its own
location, so commands work from any current directory. Python 3.11 or later;
Linux (the farm) is supported.

## Run on the farm

As the account that owns the analysis output folders (currently `xlzd`):

```bash
source /disk/groups/atp/miniconda3/etc/profile.d/conda.sh
conda activate /home/lze/nangel/.conda/envs/alpine-analysis
cd /disk/gfs_atp/xlzd/alpine/analysis
umask 0002
```

Use the existing environment if it imports all dependencies. If packages are
missing, install into the environment with its owner's permission:

```bash
python -m pip install -r Masters/Master/Python/process/requirements.txt
```

One input:

```bash
python -u Masters/Master/Python/process/process_alpine.py \
  /disk/gfs_atp/xlzd/alpine/Run3_warm/20261007T115327Z-multi-ch-10c-0-5-3-52v-1khz-30ns-2-7v-11d041c1/waveforms-0000.root
```

All ROOT files in both Run3 folders, recursively and sequentially:

```bash
python -u Masters/Master/Python/process/process_alpine.py \
  /disk/gfs_atp/xlzd/alpine/Run3_warm \
  /disk/gfs_atp/xlzd/alpine/Run3_cold \
  --skip-existing
```

To keep a terminal log without hiding a failing exit status (Bash):

```bash
mkdir -p logs
set -o pipefail
python -u Masters/Master/Python/process/process_alpine.py \
  /disk/gfs_atp/xlzd/alpine/Run3_warm --skip-existing \
  2>&1 | tee "logs/processing-$(date +%Y%m%d-%H%M%S).log"
```

## Outputs and existing files

The default directories are always the shared analysis folders, not a home:

- `/disk/gfs_atp/xlzd/alpine/analysis/parquets/<campaign>/`
- `/disk/gfs_atp/xlzd/alpine/analysis/plots/<campaign>/`

For example `<campaign>` is `Run3_warm` or `Run3_cold`. Each file produces:

- `<run-folder>__<root-stem>__<source-path-hash>.parquet`: per-event features.
- Same stem + `.pdf`: one channel per row, four channels per page by default.
- Same stem + `.summary.csv`: one channel per row, ready for later comparisons.
- Same stem + `.processing.json`: source, settings, fit parameters, provenance
  and a completion marker.

The source hash prevents collisions even when different folders contain identical
run and ROOT filenames. New version 2 output names/layout do not overwrite older
flat-folder version 1 results. Moving a source to another absolute path changes
its output hash.

Without a flag, any existing output refuses reprocessing. `--skip-existing`
skips only a complete set whose source path/size/modification time, processor,
helper, settings and recorded output sizes still match. If outputs are partial
or differ, it reports a failure requiring `--overwrite`; it never treats them
as complete. `--overwrite` replaces outputs only for the specified file(s).
Remove `--skip-existing` when using `--overwrite` (they are mutually exclusive).

Waveform processing, plots and summaries are staged before publication. The
manifest is published last. Individual renames are atomic; the set of four is
not a filesystem transaction. A failed overwrite before publication leaves old
outputs intact; an interruption during publication leaves an incomplete set,
which is detected on the next run. An operating-system advisory lock prevents
concurrent writers and is released on process exit, including crashes. The
small `.processing.lock` file intentionally remains; its presence alone does
not indicate an active process. Do not delete it while processes may use it.
Final outputs have mode 0664; lock files 0660. Output directories must already
be writable by the processing account. The script does not change ownership or
permissions on the rest of ALPINE.

## Reconstruction: same quantities and boundaries as the new helper

The active `func_notebook.analyse_root_data()` is the reference for the copied
chunk-level reconstruction code. Its hard-coded private output directory is
avoided by writing the feature table inside this script. No helper file is edited.

For every waveform:

1. Baseline is the mean of the baseline interval.
2. Noise is the population standard deviation of that interval (`ddof=0`).
3. Find the largest absolute baseline-subtracted excursion in the search interval.
4. Identify its sign, then find the last baseline crossing before the peak and
   the first crossing afterwards, searching the whole record.
5. Sum the baseline-subtracted samples between these crossings, **including both
   crossing samples**, preserving the sign of the pulse area.

No-crossing boundaries fall back to the beginning/end of the record and are
flagged. The helper's right sentinel `n_samples` is stored as `n_samples-1`, the
last actual sample; this preserves the identical sum. There is no pulse polarity
flip, baseline-sigma normalization, pulse-area cut, or deletion of flagged events.

Default `--device 64` windows are read from the unchanged helper:

- Baseline `[0,530)`; peak **search** `[564,670)`.
- `--device 4`: baseline `[0,700)`; peak search `[760,850)`.

A `sgl-ch` acquisition can still use the 64-channel digitizer. **Do not infer the
preset from the number of active channels.** Confirm search-window alignment in
the examples for each acquisition setup. Override explicitly if necessary:

```bash
python process_alpine.py /path/to/file.root \
  --baseline-window 0 530 --peak-window 564 670
```

The search window selects the pulse; the actual integration bounds are the
baseline crossings, which can extend outside that search window. This is a
change from the older version 1 fixed-window processor, matching the new helper.

## Plot grid and units

Each row is one channel. Columns, in order:

1. Signed pulse area and finger fit (if attempted), including fit status/gain.
2. Baseline noise standard deviation in **ADC**.
3. Baseline level in **ADC**.
4. Baseline vs signed area.
5. Noise vs signed area.
6. Baseline vs noise.
7. Five pulse examples by default; `--examples 8` or `--examples 10` adds more.

Examples are seeded random ROOT entries without replacement, reproducible per
channel; fewer are shown if the file contains fewer entries. They show the raw
ADC trace, baseline, baseline +/- noise in ADC, search interval, peak location,
actual inclusive integration boundaries and the entire signed integration area.
The annotations come from the same saved event features, not another independent
boundary calculation. Examples include flagged events, allowing faults to be seen.
A wide canvas is deliberate; zoom into panels. `--channels-per-page 0` puts all
channels on one page. PDF count densities are rasterized; labels/curves remain
vector graphics.

"Variance in ADC" is dimensionally a standard deviation. The plotted and summary
noise is `std_dev` in ADC; actual `noise_variance = std_dev**2` is also saved, in
ADC^2. Area and gain are **ADC-samples**, not ADC, charge, electrons, or normalized
photoelectron units. Sampling periods, electronics gain, impedance and ADC
calibration matter for physical comparisons across digitizers/settings. Provide
`--sample-period-ns` only when known; the `30ns` folder token is the LED pulse
width, not a sampling period. No charge conversion is guessed.

## Finger fitting and quality status

The script directly calls the helper's `fit_single_channel_data()` and keeps its
model, weighting, binning and optimizer. Default reduced-chi-square limit is
**2**, matching `Run3_cold`'s master processing, rather than the old limit of 100.
`--bins`, `--num-peaks` and `--max-reduced-chi2` remain configurable.

Negative pulse spectra are reflected for fitting only. `peak_integral` remains
signed; `area_for_fit` and `fit_polarity` record the orientation separately.
Auto orientation requires >=80% sign agreement among well-contained SNR>=5 peaks.
Otherwise the fit is skipped. `--polarity positive` or `--polarity negative`
sets a known orientation explicitly. It does not bypass other fit checks.

Fits use the low-SNR pedestal plus valid events with the dominant pulse sign;
zero-noise, saturated (when rails specified), missing-crossing and search-edge
peaks are excluded. All of these events still remain in the parquet and global
histograms/correlations. A blue stepped histogram shows the exact fit-subset
counts/bin edges alongside the all-events histogram, so model heights are checked
against the data actually fitted.

Additional **diagnostic, heuristic** gain screening requires:

- At least 100 fit events and three prominent observed area-spectrum maxima
  (smoothed histogram, prominence >= max(5 counts, 5% of smoothed maximum)).
- Finite fit/covariance, positive gain and finite positive gain uncertainty;
  relative gain uncertainty <=20% and reduced chi-square <= configured limit.
- At least three consecutive fitted photon indices supported by observed maxima
  within 20% of gain and with non-negligible fitted amplitudes.
- Separation exceeding twice the fitted width at the third supported peak.

These gates prevent common noise-only/unsupported fits from being used blindly;
they are not a statistical proof of calibration validity. Review accepted fits
visually before quantitative use, especially at low occupancy or with few
resolved peaks. Legitimate one/two-peak data can be marked `unresolved_peaks` and
need a separately constrained fit later. The helper still chooses its original
automatic initial anchors; no new optimizer or calibration model is introduced.
Rejected candidate curves are shown when available but **usable `gain` and
`gain_error` stay null**. Candidate gain/parameters/covariance, rejection reasons
and chi-square live in JSON/Parquet metadata. Rejected channels do not fail the
file's reconstruction; their status is shown in the PDF and summary.

## Parquet and comparison summary

Original four feature columns remain: `channel`, `baseline`, `std_dev`,
`peak_integral`. Added columns include:

- `event_index`: zero-based ROOT entry within each channel, independent of chunks.
- `noise_variance`, `baseline_delta_adc` (second-half minus first-half baseline).
- `peak_index`, signed `peak_amplitude_adc`, event `polarity`, `peak_snr`.
- Inclusive `integration_start`, `integration_end`; `record_samples`.
- `left_crossing_found`, `right_crossing_found`, `peak_at_search_edge`, `noise_zero`,
  `low_snr` (SNR<5 or undefined), and `saturated`.
- Channel-level `gain`, `gain_error`, `fit_status`, `fit_polarity`, `area_for_fit`.
- ROOT scalar branches `event`, `timestamp_ticks`, `timestamp_ns`, `trigger_id`,
  `sample_period_ns`, **if present**. Timestamp ticks are preserved without
  inventing their clock conversion.

Saturation is checked only with explicitly supplied `--adc-min` and/or
`--adc-max`; uint16 storage alone does not identify the hardware ADC limits.
Without rails, `saturated` is false and `saturation_checked` in metadata/summary
is false, meaning **not checked**. Missing crossings, low SNR and search-edge
flags are diagnostics, not automatic event cuts in the saved data.

Only example waveforms are kept temporarily in memory; no waveform vectors are
saved in parquet, matching the active chunked helper. The original ROOT remains
the source for future waveform-level analyses. Processing supports `Events`
TTrees and RNTuples with `chNN` waveform branches. Nonfinite samples, unequal
lengths inside a chunk, empty input or invalid/short records fail that file.
Other files in a batch continue; any file failure produces a nonzero batch exit.

The `.summary.csv` contains one row per channel: gain/status, mean/median
baseline and noise, baseline spread, signed-area statistics, flag fractions,
experimental conditions and source. It supports the next comparison step without
scanning waveform records. Folder parsing reuses the helper, additionally handling
`sgl-ch`, `1khz` and hyphenated LED decimals. The helper's temperature convention
(unsigned values >40 interpreted as negative) is retained and recorded explicitly;
raw helper metadata is preserved separately.

Processing settings, source signature, helper/processor hashes, example entries
and full fits are also JSON in Parquet schema metadata:

```python
import json
import pyarrow.parquet as pq
info = json.loads(pq.read_metadata("file.parquet").metadata[b"alpine_processing"])
```

## Dark counts and physical limits

`dark_count_rate_hz` is intentionally blank. This algorithm selects **one pulse
inside a trigger search window**, including the largest noise excursion when
there is no real pulse. It is not an unbiased pulse counter. The baseline-crossing
rule can truncate noisy tails, and overlapping pulses may merge before crossing;
these are properties of the existing reconstruction. Quantitative dark-count
rates need separately defined off-trigger windows, pulse finding/efficiency,
thresholds, known sample periods and effective live time. ROOT timestamps are
retained to help that future work but do not alone supply the required exposure.

Waveform RAM is bounded by `--chunk-events` (default 2000) for one channel at a
time. Scalar features for one channel are loaded for fitting/plotting; for huge
files that table and the fit can still be expensive. Source must be fully written
before processing; its size/mtime are checked before publishing outputs.

## Verification performed for this refresh

Synthetic tests checked exact agreement with the active helper for positive,
negative and noise-only waveforms, plus the 4-channel preset; independent
crossing/sum calculations; preserved entry indices/timing across chunks; noise
units; recovery of an injected 192 ADC-sample gain for both pulse signs; rejection
of noise-only gain; TTree/RNTuple inputs; short/nonfinite records; boundary and
saturation flags; overwrite protection, settings-aware skip and active locking.
The wide multi-page PDF was rendered and visually inspected. The uploaded archive
contains no acquisition ROOT files, so real-data validation still requires checking
your first Run3 PDF, search-window alignment and fitted spectra on the farm.
