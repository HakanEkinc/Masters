#!/usr/bin/env python3
"""Non-interactive, one-ROOT-file ALPINE reconstruction and PDF report.

Place beside the supplied func_notebook.py (unchanged), then run:
    python process_alpine.py /path/to/run/waveforms-0000.root
    python process_alpine.py /path/to/run/waveforms-0000.root --overwrite
    python process_alpine.py /path/to/run/waveforms-0000.root --examples 8

Dependencies: numpy pandas scipy matplotlib seaborn uproot awkward pyarrow
    python -m pip install numpy pandas scipy matplotlib seaborn uproot awkward pyarrow

Defaults: /disk/gfs_atp/xlzd/alpine/analysis/{parquets,plots}.
Outputs: <run>__<root-stem>.parquet and <run>__<root-stem>.pdf.
Use --parquet-dir and --plot-dir to change destinations. Both existing outputs
are protected unless --overwrite is supplied. A lock prevents concurrent runs
for the same parquet. Temporary outputs are built before publishing; each rename
is atomic, but the two-file publication is not a single filesystem transaction.

Reconstruction uses nbk.analyse_root_data unchanged: baseline samples [0,530),
peak samples [564,670), population noise (ddof=0), channel-wide polarity flip,
and a signed sum over ALL baseline-subtracted peak samples. Windows are read
from the helper, not duplicated here. No sampling period, impedance or ADC
conversion is supplied, so area and fitted gain are in ADC-samples, not charge.

Original parquet columns are preserved. Added event_index is the zero-based
ROOT entry within each channel. polarity is +1/-1; gain, gain_error and
fit_status are channel-level values repeated on events. Failed fits retain data
with null gains. Full returned fit parameters/covariance/peak positions, source,
windows, helper hash and example entry indices live in Parquet schema metadata
under b'alpine_processing' (JSON; inspect with pyarrow.parquet.read_metadata).
No raw waveforms were saved by the notebook, and none are added here.

The PDF has one channel per row: baseline, noise, area with the existing gain
fit, the notebook's three correlations, and 5-10 annotated pulse examples.
It is intentionally wide: zoom to inspect individual panels. Default pagination
is four channels per page; --channels-per-page 0 puts all channels on one page.
Examples are seeded random entries without replacement (fewer if unavailable).

The notebook gain fit and its chi-square threshold of 100 are retained, not
revalidated as a calibration method. Fit failures do not abort reconstruction.
Malformed/empty/nonfinite or too-short waveform input does abort. Like the
notebook, reconstruction loads the whole ROOT file into memory; this is not a
streaming processor. Folder metadata is stored as parsed, not interpreted (the
helper's temperature sign parsing can be ambiguous). No automation is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root_file', type=Path)
    p.add_argument('--parquet-dir', type=Path, default=Path('/disk/gfs_atp/xlzd/alpine/analysis/parquets'))
    p.add_argument('--plot-dir', type=Path, default=Path('/disk/gfs_atp/xlzd/alpine/analysis/plots'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--examples', type=int, choices=range(5, 11), default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--channels-per-page', type=int, default=4)
    p.add_argument('--bins', type=int, default=500)
    p.add_argument('--num-peaks', type=int, default=18)
    p.add_argument('--max-reduced-chi2', type=float, default=100.0)
    a = p.parse_args()
    if a.channels_per_page < 0 or a.num_peaks < 1 or a.bins <= 7 + a.num_peaks or not 0 < a.max_reduced_chi2 < float('inf') or a.seed < 0:
        p.error('Require nonnegative page size/seed, positive peaks/chi2, and bins > 7 + num-peaks.')
    return a


def clean_json(value):
    """Convert arrays and nonfinite fit entries into portable JSON."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_channel(tree, ch, minimum_length):
    import numpy as np
    w = tree[ch].array(library='np')
    if w.dtype == object:
        try:
            w = np.vstack(w)
        except ValueError as exc:
            raise ValueError(f'{ch}: unequal waveform lengths') from exc
    if w.ndim != 2 or not len(w) or w.shape[1] < minimum_length:
        raise ValueError(f'{ch}: empty/invalid waveforms or fewer than {minimum_length} samples')
    if not np.isfinite(w).all():
        raise ValueError(f'{ch}: nonfinite waveform samples; refusing silent filtering')
    return w


def report(df, root, destination, nbk, args, windows):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import uproot
    channels = sorted(df.channel.unique(), key=lambda c: int(c[2:]))
    per_page = args.channels_per_page or len(channels)
    fits, examples = {}, {}
    rng = np.random.default_rng(args.seed)
    bs, be, ps, pe = windows
    labels = {'baseline': 'Baseline [ADC]', 'std_dev': 'Noise σ [ADC]', 'peak_integral': 'Area [ADC-samples]'}
    pairs = [('baseline', 'peak_integral'), ('baseline', 'std_dev'), ('std_dev', 'peak_integral')]
    with uproot.open(root) as f, PdfPages(destination) as pdf:
        pdf.infodict()['Title'] = f'ALPINE reconstruction: {root.parent.name}/{root.name}'
        for offset in range(0, len(channels), per_page):
            group = channels[offset:offset + per_page]
            fig, axes = plt.subplots(len(group), 6 + args.examples, figsize=(4.0*(6+args.examples), 3.7*len(group)), squeeze=False)
            try:
                for row, ch in enumerate(group):
                    d = df.loc[df.channel == ch]
                    ax = axes[row]
                    for col, key in enumerate(('baseline', 'std_dev')):
                        ax[col].hist(d[key], bins=args.bins, color='steelblue')
                        ax[col].set(title=f'{ch}: {key} (N={len(d)})', xlabel=labels[key], ylabel='Events')
                    fit = {'status': 'failed_or_rejected', 'gain': None, 'gain_error': None}
                    try:
                        popt, pcov, peaks = nbk.fit_pandas_data(d.peak_integral, channel_name=ch, ax=ax[2], bins=args.bins, num_peaks_to_fit=args.num_peaks, max_reduced_chi2=args.max_reduced_chi2)
                        if popt is not None and np.isfinite(popt).all() and popt[4] > 0:
                            err = float(np.sqrt(pcov[4, 4])) if pcov is not None and pcov[4, 4] >= 0 else float('nan')
                            fit.update(status='accepted_by_notebook', gain=float(popt[4]), gain_error=err, raw_popt=popt, raw_pcov=pcov, peak_params=peaks, base_mu_0=popt[2], base_sigma_0=popt[3], base_sigma_1=popt[5])
                            ax[2].text(.98, .97, f'Gain={popt[4]:.4g} ± {err:.2g}', transform=ax[2].transAxes, ha='right', va='top', fontsize=8)
                    except Exception as exc:
                        fit['error'] = str(exc)
                        ax[2].clear()
                        ax[2].hist(d.peak_integral, bins=args.bins)
                        ax[2].set_title(f'{ch}: fit failed (see metadata)')
                    if fit['status'] != 'accepted_by_notebook':
                        print(f'WARNING: {ch}: gain fit failed/rejected; keeping reconstructed events.', file=sys.stderr)
                    ax[2].set_xlabel(labels['peak_integral'])
                    ax[2].set_ylabel('Events')
                    ax[2].legend(fontsize=6, loc='upper left')
                    fits[ch] = fit
                    for col, (x, y) in enumerate(pairs, 3):
                        hb = ax[col].hexbin(d[x], d[y], gridsize=50, mincnt=1, cmap='viridis', rasterized=True)
                        fig.colorbar(hb, ax=ax[col], label='Events')
                        ax[col].set(title=f'{ch}: {x} vs {y}', xlabel=labels[x], ylabel=labels[y])
                    w = read_channel(f['Events'], ch, max(be, pe))
                    ids = np.sort(rng.choice(len(w), min(args.examples, len(w)), replace=False))
                    examples[ch] = ids.tolist()
                    for col in range(args.examples):
                        a = ax[6 + col]
                        if col >= len(ids):
                            a.text(.5, .5, 'No further events', ha='center', transform=a.transAxes)
                            a.set_axis_off()
                            continue
                        entry = int(ids[col])
                        event = d.iloc[entry]
                        sig = w[entry].astype(float)
                        t = np.arange(len(sig))
                        b, s = event.baseline, event.std_dev
                        a.plot(t, sig, lw=.7, color='steelblue')
                        a.axvspan(bs, be, color='green', alpha=.08, label=f'Baseline [{bs},{be})')
                        for bound in (bs, be):
                            a.axvline(bound, color='green', ls=':', lw=.8)
                        for bound in (ps, pe):
                            a.axvline(bound, color='black', ls='--', lw=.8)
                        a.axhline(b, color='green', ls='--', lw=.8, label='Baseline')
                        a.axhline(b+s, color='orange', ls=':', lw=.8, label='±1σ (ddof=0)')
                        a.axhline(b-s, color='orange', ls=':', lw=.8)
                        a.fill_between(t[ps:pe], sig[ps:pe], b, color='purple', alpha=.3, label=f'Area [{ps},{pe})')
                        a.set(title=f'{ch}: entry {entry} | polarity {int(event.polarity):+d}\nB={b:.4g}, σ={s:.3g}, area={event.peak_integral:.5g}', xlabel='Sample index', ylabel='Signal [ADC]')
                        a.legend(fontsize=6, loc='best')
                    del w
                    for a in ax:
                        a.grid(alpha=.15)
                        a.tick_params(labelsize=7)
                        a.title.set_fontsize(9)
                fig.suptitle(f'{root.parent.name}/{root.name}\nBaseline [{bs},{be}); integration [{ps},{pe}); signed sum after channel polarity correction', fontsize=12)
                fig.tight_layout(rect=(0, 0, 1, .93))
                pdf.savefig(fig, dpi=140)
            finally:
                plt.close(fig)
    return fits, examples


def main():
    args = arguments()
    root = args.root_file.expanduser().resolve(strict=True)
    if not root.is_file() or root.suffix.lower() != '.root':
        raise ValueError('Input must be a ROOT file')
    stem = f'{root.parent.name}__{root.stem}'
    pq_path = args.parquet_dir.expanduser().resolve() / f'{stem}.parquet'
    pdf_path = args.plot_dir.expanduser().resolve() / f'{stem}.pdf'
    for p in (pq_path, pdf_path):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f'{p} exists. Use --overwrite to reprocess and replace outputs.')
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import func_notebook as nbk
    windows = (nbk.baseline_start_64, nbk.baseline_end_64, nbk.peak_start_64, nbk.peak_end_64)
    bs, be, ps, pe = windows
    if not (0 <= bs < be and 0 <= ps < pe):
        raise ValueError(f'Invalid helper windows: {windows}')
    for p in (pq_path, pdf_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    lock = pq_path.with_suffix('.processing.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Processing lock exists: {lock}. Check for an active process before removing a stale lock.') from None
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f'pid={os.getpid()} source={root}\n')
        for p in (pq_path, pdf_path):
            if p.exists() and not args.overwrite:
                raise FileExistsError(f'{p} exists; use --overwrite')
        source_stat = root.stat()
        polarities = {}
        with uproot.open(root) as f:
            tree = f['Events']
            import re
            channels = [ch for ch in tree.keys() if re.fullmatch(r'ch[0-9]+', ch)]
            if not channels:
                raise ValueError('Events has no ch[0-9]+ waveform branches')
            for ch in channels:
                w = read_channel(tree, ch, max(be, pe))
                corrected = w[:, ps:pe] - np.mean(w[:, bs:be], axis=1)[:, None]
                polarities[ch] = -1 if np.mean(corrected) < 0 else 1
                del w, corrected
        with tempfile.TemporaryDirectory(prefix='.alpine-', dir=pq_path.parent) as pq_tmp, tempfile.TemporaryDirectory(prefix='.alpine-', dir=pdf_path.parent) as pdf_tmp:
            # Isolate the helper's run-only filename; originals are never modified.
            df, staged_pq = nbk.analyse_root_data(str(root), pq_tmp)
            df['event_index'] = df.groupby('channel', sort=False).cumcount()
            df['polarity'] = df.channel.map(polarities).astype('int8')
            staged_pdf = Path(pdf_tmp) / pdf_path.name
            fits, examples = report(df, root, staged_pdf, nbk, args, windows)
            for key in ('gain', 'gain_error'):
                df[key] = df.channel.map({ch: fit[key] for ch, fit in fits.items()}).astype(float)
            df['fit_status'] = df.channel.map({ch: fit['status'] for ch, fit in fits.items()})
            if (root.stat().st_size, root.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise RuntimeError('Source changed during processing; outputs not published')
            metadata = clean_json(dict(format_version=1, source=str(root), source_size=source_stat.st_size, source_mtime_ns=source_stat.st_mtime_ns, created_utc=datetime.now(timezone.utc).isoformat(), helper_sha256=hashlib.sha256(Path(nbk.__file__).read_bytes()).hexdigest(), folder_metadata_uninterpreted=nbk.parse_folder_metadata(str(root)), baseline_window=[bs, be], integration_window=[ps, pe], noise_ddof=0, area_units='ADC-samples', polarity=polarities, fits=fits, examples=examples, seed=args.seed, bins=args.bins, num_peaks=args.num_peaks, max_reduced_chi2=args.max_reduced_chi2))
            table = pa.Table.from_pandas(df, preserve_index=False)
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), b'alpine_processing': json.dumps(metadata, allow_nan=False).encode()})
            pq.write_table(table, staged_pq)
            os.replace(staged_pdf, pdf_path)
            os.replace(staged_pq, pq_path)
        print(f'Saved {len(df)} events across {len(channels)} channels\nParquet: {pq_path}\nPDF: {pdf_path}')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

