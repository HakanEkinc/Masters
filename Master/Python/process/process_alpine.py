#!/usr/bin/env python3
"""ALPINE ROOT -> per-event parquet, channel-grid PDF and channel summary.

Usage (from any directory):
  python process_alpine.py /path/to/run/waveforms-0000.root
  python process_alpine.py /disk/gfs_atp/xlzd/alpine/Run3_warm --skip-existing
  python process_alpine.py /path/to/file.root --overwrite --examples 8

The helper is imported from ../func_notebook.py relative to this script.
Baseline-crossing integration matches its active analyse_root_data function.
See readme.md for output layout, units, fit checks and limitations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap

VERSION = '2.0'
ANALYSIS = Path('/disk/gfs_atp/xlzd/alpine/analysis')
SCALARS = ('event', 'timestamp_ticks', 'timestamp_ns', 'trigger_id', 'sample_period_ns')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', type=Path, nargs='+', help='ROOT files or folders (searched recursively)')
    p.add_argument('--parquet-dir', type=Path, default=ANALYSIS / 'parquets')
    p.add_argument('--plot-dir', type=Path, default=ANALYSIS / 'plots')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--overwrite', action='store_true', help='replace outputs for selected inputs')
    g.add_argument('--skip-existing', action='store_true', help='skip only complete outputs with matching source and settings')
    p.add_argument('--device', choices=('64', '4'), default='64', help='window preset; NOT inferred from number of active channels')
    p.add_argument('--baseline-window', type=int, nargs=2, metavar=('START', 'STOP'))
    p.add_argument('--peak-window', type=int, nargs=2, metavar=('START', 'STOP'), help='peak SEARCH interval, not integration limits')
    p.add_argument('--examples', type=int, choices=range(5, 11), default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--channels-per-page', type=int, default=4, help='0 = all channels on one page')
    p.add_argument('--chunk-events', type=int, default=2000)
    p.add_argument('--bins', type=int, default=500)
    p.add_argument('--num-peaks', type=int, default=18)
    p.add_argument('--max-reduced-chi2', type=float, default=2.0, help='same default as Run3 master_function_root')
    p.add_argument('--polarity', choices=('auto', 'positive', 'negative'), default='auto', help='orientation for fit only; saved peak_integral stays signed')
    p.add_argument('--sample-period-ns', type=float, help='known waveform sampling period; never inferred from LED width')
    p.add_argument('--adc-min', type=float, help='known ADC lower rail for saturation flag')
    p.add_argument('--adc-max', type=float, help='known ADC upper rail for saturation flag')
    a = p.parse_args()
    if a.seed < 0 or a.channels_per_page < 0 or a.chunk_events < 1 or a.num_peaks < 1 or a.bins <= 7+a.num_peaks:
        p.error('Require nonnegative seed/page size, positive chunk size/peaks, bins > 7 + num-peaks')
    import math
    for key in ('max_reduced_chi2', 'sample_period_ns'):
        v = getattr(a, key)
        if v is not None and (not math.isfinite(v) or v <= 0):
            p.error(f'{key} must be finite and positive')
    for key in ('adc_min', 'adc_max'):
        v = getattr(a, key)
        if v is not None and not math.isfinite(v):
            p.error(f'{key} must be finite')
    if a.adc_min is not None and a.adc_max is not None and a.adc_min >= a.adc_max:
        p.error('adc-min must be smaller than adc-max')
    return a


def load_helper():
    # Resolve ../func_notebook.py from this file, not the shell working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import matplotlib
    matplotlib.use('Agg')
    import func_notebook
    return func_notebook


def windows_from_helper(args, nbk):
    suffix = args.device
    bs, be = args.baseline_window or (getattr(nbk, f'baseline_start_{suffix}'), getattr(nbk, f'baseline_end_{suffix}'))
    ps, pe = args.peak_window or (getattr(nbk, f'peak_start_{suffix}'), getattr(nbk, f'peak_end_{suffix}'))
    if not (0 <= bs < be <= ps < pe):
        raise ValueError('Require nonempty, nonoverlapping baseline/search windows: 0 <= baseline start < stop <= peak start < stop')
    return bs, be, ps, pe


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def reconstruct(waveforms, windows, adc_min=None, adc_max=None):
    """Active helper's baseline/crossing/signed integration code, plus saved bounds.

    Crossings are inclusive; if absent, use 0 or the final recorded sample.
    The helper's right sentinel is n_samples; clipping it to n_samples-1 here
    preserves exactly the same sum and gives an actual, plottable index.
    No sign flip, threshold-based integration, filtering or pulse rejection.
    """
    import numpy as np
    w = np.asarray(waveforms)
    if w.dtype == object:
        try:
            w = np.vstack(w)
        except ValueError as e:
            raise ValueError('Unequal waveform lengths inside a chunk') from e
    bs, be, ps, pe = windows
    if w.ndim != 2 or not len(w) or w.shape[1] < pe:
        raise ValueError(f'Waveforms must be a nonempty matrix with at least {pe} samples')
    if not np.isfinite(w).all():
        raise ValueError('Nonfinite waveform samples; refusing to silently delete samples')
    baseline_window = w[:, bs:be]
    baselines = np.mean(baseline_window, axis=1)
    std_devs = np.std(baseline_window, axis=1)
    corrected_waveforms = w - baselines[:, None]
    corrected_peak_window = corrected_waveforms[:, ps:pe]
    peaks = np.argmax(np.abs(corrected_peak_window), axis=1) + ps
    n_samples = w.shape[1]
    indices = np.arange(n_samples)
    peak_values = corrected_waveforms[np.arange(len(w)), peaks]
    peak_signs = np.sign(peak_values)
    peak_signs[peak_signs == 0] = 1
    aligned_waveforms = corrected_waveforms * peak_signs[:, None]
    below_baseline = aligned_waveforms <= 0
    left_valid = below_baseline & (indices < peaks[:, None])
    left_crossings = np.max(np.where(left_valid, indices, 0), axis=1)
    left_found = np.any(left_valid, axis=1)
    right_valid = below_baseline & (indices > peaks[:, None])
    right_crossings = np.min(np.where(right_valid, indices, n_samples), axis=1)
    right_found = np.any(right_valid, axis=1)
    integration_mask = (indices >= left_crossings[:, None]) & (indices <= right_crossings[:, None])
    integrals = np.sum(corrected_waveforms * integration_mask, axis=1)
    snr = np.divide(np.abs(peak_values), std_devs, out=np.full(len(w), np.nan), where=std_devs > 0)
    saturated = np.zeros(len(w), dtype=bool)
    if adc_min is not None:
        saturated |= np.any(w <= adc_min, axis=1)
    if adc_max is not None:
        saturated |= np.any(w >= adc_max, axis=1)
    half = max(1, (be-bs)//2)
    baseline_delta = np.mean(baseline_window[:, half:], axis=1) - np.mean(baseline_window[:, :half], axis=1) if be-bs > 1 else np.zeros(len(w))
    return {
        'baseline': baselines, 'std_dev': std_devs, 'noise_variance': std_devs**2,
        'peak_integral': integrals, 'peak_index': peaks.astype('int32'),
        'peak_amplitude_adc': peak_values, 'polarity': peak_signs.astype('int8'),
        'integration_start': left_crossings.astype('int32'),
        'integration_end': np.minimum(right_crossings, n_samples-1).astype('int32'),
        'left_crossing_found': left_found, 'right_crossing_found': right_found,
        'peak_at_search_edge': (peaks == ps) | (peaks == pe-1),
        'noise_zero': std_devs == 0, 'peak_snr': snr,
        'low_snr': (snr < 5) | ~np.isfinite(snr),
        'saturated': saturated, 'baseline_delta_adc': baseline_delta,
        'record_samples': np.full(len(w), n_samples, dtype='int32'),
    }


def normalized_metadata(root, nbk):
    # Preserve helper interpretation separately; repair only unambiguous tokens.
    raw = nbk.parse_folder_metadata(str(root))
    normalized = dict(raw)
    name = root.parent.name
    ch = re.search(r'(sgl-ch|single-ch|multi-ch)', name, re.I)
    if ch:
        normalized['channel_type'] = ch.group(1).lower()
    hz = re.search(r'-(\d+(?:\.\d+)?)(k?hz)-', name, re.I)
    if hz:
        normalized['frequency_hz'] = float(hz.group(1)) * (1000 if hz.group(2).lower().startswith('k') else 1)
    led = re.search(r'-\d+(?:\.\d+)?ns-(\d+(?:[-.]\d+)?)v(?:-|$)', name, re.I)
    if led:
        normalized['led_voltage'] = float(led.group(1).replace('-', '.'))
        normalized['pulse_voltage'] = normalized['led_voltage']
    normalized['temperature_convention'] = 'helper convention: unsigned values above 40 C interpreted as negative'
    return raw, normalized


def source_signature(root):
    s = root.stat()
    return {'path': str(root), 'size_bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def campaign_name(root):
    for parent in root.parents:
        if re.fullmatch(r'Run\d+(?:_.*)?|run1_2|shape_test|Warm_after_ampGND', parent.name, re.I):
            return parent.name
    return root.parent.parent.name or 'data'


def output_paths(root, args):
    campaign = campaign_name(root)
    # Full source path hash also disambiguates repeated folder/file names.
    uid = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    stem = f'{root.parent.name}__{root.stem}__{uid}'
    pd = args.parquet_dir.expanduser().resolve() / campaign
    vd = args.plot_dir.expanduser().resolve() / campaign
    return {'parquet': pd / f'{stem}.parquet', 'pdf': vd / f'{stem}.pdf',
            'summary': vd / f'{stem}.summary.csv', 'manifest': vd / f'{stem}.processing.json'}


@contextmanager
def processing_lock(path):
    # flock is released by the OS on process exit. Keep the inode: deleting a
    # lock pathname can let different processes lock different inodes.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o660)
    try:
        os.fchmod(fd, 0o660)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'Another process is using this output: {path}') from e
        os.ftruncate(fd, 0)
        os.write(fd, f'pid={os.getpid()}\n'.encode())
        yield
    finally:
        os.close(fd)


def extract_features(root, stage, nbk, args, windows):
    """Read one channel in bounded waveform chunks; retain only example traces."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot
    examples, records = {}, {}
    writer = None
    with uproot.open(root) as f:
        if 'Events' not in f:
            raise ValueError('ROOT file has no Events tree')
        tree = f['Events']
        channels = sorted([k for k in tree.keys() if re.fullmatch(r'ch\d+', k)], key=lambda x: int(x[2:]))
        if not channels or not tree.num_entries:
            raise ValueError('Events is empty or has no ch[0-9]+ waveform branches')
        scalars = [k for k in SCALARS if k in tree.keys()]
        try:
            for ch in channels:
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(ch[2:])]))
                selected = np.sort(rng.choice(tree.num_entries, min(args.examples, tree.num_entries), replace=False))
                examples[ch] = {}
                offset = 0
                print(f'  Reconstructing {ch}: {tree.num_entries} entries', flush=True)
                for arrays in tree.iterate(expressions=[ch] + scalars, step_size=args.chunk_events, library='np', how=dict):
                    # Normalize NumPy structured-array output used by some RNTuples.
                    if not isinstance(arrays, dict):
                        arrays = {k: arrays[k] for k in arrays.dtype.names}
                    w = arrays[ch]
                    features = reconstruct(w, windows, args.adc_min, args.adc_max)
                    n = len(features['baseline'])
                    df = pd.DataFrame({'channel': np.repeat(ch, n), 'event_index': np.arange(offset, offset+n, dtype='int64'), **features})
                    for key in scalars:
                        values = np.asarray(arrays[key])
                        if values.ndim != 1 or len(values) != n or values.dtype.kind not in 'biuf':
                            raise ValueError(f'{key} is not a numeric scalar per ROOT entry')
                        df[key] = values
                    for entry in selected[(selected >= offset) & (selected < offset+n)]:
                        local = int(entry-offset)
                        examples[ch][int(entry)] = (np.asarray(w[local], dtype=float).copy(), df.iloc[local].to_dict())
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(stage, table.schema, compression='zstd')
                    writer.write_table(table)
                    offset += n
                if offset != tree.num_entries:
                    raise ValueError(f'{ch}: read {offset} entries, expected {tree.num_entries}')
                records[ch] = offset
        finally:
            if writer is not None:
                writer.close()
    return channels, examples, records, scalars


def stable_model(x, *params):
    """Identical helper model, evaluating shifted exponential only on its domain."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    a, decay, mu, s0, gain, s1, const = params[:7]
    y = np.full_like(x, const)
    mask = x >= mu
    y[mask] += a * np.exp(-decay * (x[mask]-mu))
    for n, height in enumerate(params[7:]):
        width = np.sqrt(s0*s0 + n*s1*s1)
        y += height * np.exp(-0.5*((x-mu-n*gain)/width)**2)
    return y


def fit_channel(df, ch, nbk, args):
    """Reuse helper fitter; screen fit support before publishing a usable gain."""
    import numpy as np
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    out = {'status': 'not_fitted', 'gain': None, 'gain_error': None, 'candidate_gain': None,
           'fit_polarity': None, 'reasons': [], 'raw_popt': None, 'raw_pcov': None,
           'peak_params': None, 'reduced_chi2': None, 'n_fit_events': 0}
    eligible = (~df.low_snr & ~df.noise_zero & ~df.saturated & df.left_crossing_found & df.right_crossing_found & ~df.peak_at_search_edge)
    strong = df.loc[eligible, 'polarity'].to_numpy()
    if args.polarity != 'auto':
        sign = 1 if args.polarity == 'positive' else -1
    elif not len(strong) or max(np.mean(strong > 0), np.mean(strong < 0)) < .8:
        out.update(status='ambiguous_polarity', reasons=['No consistent polarity among well-contained peaks with SNR >= 5'])
        return out
    else:
        sign = 1 if np.mean(strong > 0) >= .8 else -1
    out['fit_polarity'] = sign
    # Keep the low-SNR pedestal in the fit; exclude truncated/search-edge/saturated
    # events and high-SNR pulses with opposite polarity. All events remain saved.
    use = df.left_crossing_found & df.right_crossing_found & ~df.peak_at_search_edge & ~df.saturated & ~df.noise_zero
    use &= df.low_snr | (df.polarity == sign)
    raw = df.loc[use, 'peak_integral'].to_numpy(dtype=float) * sign
    out['n_fit_events'] = len(raw)
    if len(raw) < 100 or np.ptp(raw) <= 0:
        out.update(status='insufficient_data', reasons=['Fewer than 100 fit events or no area spread'])
        return out
    counts, edges = np.histogram(raw, bins=args.bins)
    centers = (edges[:-1]+edges[1:])/2
    smooth = gaussian_filter1d(counts.astype(float), 1)
    observed, _ = find_peaks(smooth, prominence=max(5., .05*float(smooth.max())), distance=4)
    out['observed_peak_centers'] = centers[observed]
    if len(observed) < 3:
        out.update(status='unresolved_peaks', reasons=['Fewer than three prominent observed area peaks'])
        return out
    print(f'  Fitting {ch}: {len(raw)} events, {len(observed)} prominent peaks', flush=True)
    try:
        # Original helper's np.where evaluates a discarded exponential branch;
        # suppress those overflow warnings, then check the actual output below.
        with np.errstate(over='ignore', invalid='ignore'):
            fit = nbk.fit_single_channel_data(pd.Series(raw), ch, bins=args.bins, num_peaks_to_fit=args.num_peaks, max_reduced_chi2=args.max_reduced_chi2)
    except Exception as e:
        out.update(status='fit_failed', reasons=[str(e)])
        return out
    out['reduced_chi2'] = fit.get('reduced_chi_square')
    out['ndf'] = fit.get('ndf')
    if fit.get('popt') is None:
        out.update(status='fit_failed_or_chi2_rejected', reasons=['Helper failed to converge or rejected its chi-square'])
        return out
    p, cov = np.asarray(fit['popt']), np.asarray(fit['pcov'])
    out.update(raw_popt=p, raw_pcov=cov, peak_params=fit['peak_params'], candidate_gain=float(p[4]),
               fit_histogram_edges=fit['bin_edges'], fit_histogram_counts=fit['counts'])
    reasons = []
    if not np.isfinite(p).all() or p[4] <= 0 or not np.isfinite(stable_model(centers, *p)).all():
        out.update(status='quality_rejected', reasons=['Nonfinite parameters/model or nonpositive gain'])
        return out
    error = float(np.sqrt(cov[4, 4])) if np.isfinite(cov).all() and cov[4, 4] > 0 else float('nan')
    out['candidate_gain_error'] = error
    if not np.isfinite(error) or error/p[4] > .2:
        reasons.append('Invalid covariance or relative gain uncertainty > 20%')
    if fit['ndf'] <= 0 or not np.isfinite(out['reduced_chi2']) or out['reduced_chi2'] > args.max_reduced_chi2:
        reasons.append('Invalid or excessive reduced chi-square')
    # Require independent observed peaks to map to consecutive fitted photon
    # indices within 20% of gain; artificial fine-spaced Gaussian combs fail.
    assignments = np.rint((centers[observed]-p[2])/p[4]).astype(int)
    supported = {int(n) for x, n in zip(centers[observed], assignments)
                 if 0 <= n < len(p)-7 and abs(x-(p[2]+n*p[4])) <= .2*p[4]
                 and p[7+n] >= .01*counts.max()}
    out['supported_peak_indices'] = sorted(supported)
    triples = [n for n in supported if n+1 in supported and n+2 in supported]
    if not triples:
        reasons.append('No three consecutive fitted peaks supported by observed maxima')
    elif not any(p[4] > 2*np.sqrt(p[3]**2+(n+2)*p[5]**2) for n in triples):
        reasons.append('Supported photon peaks are not resolved relative to fitted widths')
    if reasons:
        out.update(status='quality_rejected', reasons=reasons)
    else:
        out.update(status='accepted_diagnostic', gain=float(p[4]), gain_error=error)
    return out


def plot_pulse(ax, waveform, event, windows):
    import numpy as np
    bs, be, ps, pe = windows
    baseline, noise = event['baseline'], event['std_dev']
    left, right, peak = (int(event[k]) for k in ('integration_start', 'integration_end', 'peak_index'))
    t = np.arange(len(waveform))
    ax.plot(t, waveform, color='#2b6b95', lw=.65)
    ax.axvspan(bs, be, color='#42965c', alpha=.09, label=f'Baseline [{bs},{be})')
    ax.axhline(baseline, color='#31824a', ls='--', lw=.9, label=f'B = {baseline:.4g} ADC')
    ax.axhline(baseline+noise, color='#d39321', ls=':', lw=.75, label=f'Noise = {noise:.3g} ADC')
    ax.axhline(baseline-noise, color='#d39321', ls=':', lw=.75)
    for bound in (ps, pe):
        ax.axvline(bound, color='#777777', ls=':', lw=.8)
    ax.axvspan(ps, pe, color='#777777', alpha=.04, label=f'Search [{ps},{pe})')
    for bound in (left, right):
        ax.axvline(bound, color='#7a3689', ls='--', lw=1)
    # Include every sample in the saved signed sum, including crossing samples.
    ax.fill_between(t[left:right+1], waveform[left:right+1], baseline, color='#9951a2', alpha=.25,
                    label=f'Sum [{left},{right}] inclusive')
    ax.plot(peak, waveform[peak], 'o', color='#a62a2a', ms=3, label=f'Peak {peak}')
    flags = [k for k in ('low_snr', 'noise_zero', 'saturated', 'peak_at_search_edge') if event[k]]
    if not event['left_crossing_found'] or not event['right_crossing_found']:
        flags.append('missing crossing')
    ax.set_title(f"{event['channel']} / entry {int(event['event_index'])}\nArea = {event['peak_integral']:.5g} ADC-samples" + ('\n'+', '.join(flags) if flags else ''), fontsize=8)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Signal [ADC]')
    ax.legend(fontsize=5.5, loc='best')


def make_report(stage, destination, channels, examples, nbk, args, windows, metadata):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fits, summaries = {}, []
    per_page = args.channels_per_page or len(channels)
    labels = {'peak_integral': 'Signed area [ADC-samples]', 'std_dev': 'Baseline noise SD [ADC]', 'baseline': 'Baseline [ADC]'}
    pairs = [('baseline', 'peak_integral'), ('std_dev', 'peak_integral'), ('baseline', 'std_dev')]
    with PdfPages(destination) as pdf:
        pdf.infodict()['Title'] = 'ALPINE: '+metadata['source']['path']
        for offset in range(0, len(channels), per_page):
            group = channels[offset:offset+per_page]
            fig, axes = plt.subplots(len(group), 6+args.examples, figsize=(3.9*(6+args.examples), 4.0*len(group)), squeeze=False)
            try:
                for row, ch in enumerate(group):
                    # Only scalar features for this channel are loaded for fitting.
                    df = pq.read_table(stage, filters=[('channel', '=', ch)]).to_pandas()
                    fit = fit_channel(df, ch, nbk, args)
                    fits[ch] = fit
                    a = axes[row]
                    for col, key in enumerate(('peak_integral', 'std_dev', 'baseline')):
                        a[col].hist(df[key], bins=args.bins, color='#2b6b95' if col else '#a3abb3', alpha=.65, label='All saved events')
                        a[col].set(xlabel=labels[key], ylabel='Events', title=f'{ch}: '+('Area / finger fit' if col == 0 else ('Noise in ADC' if col == 1 else 'Baseline in ADC')))
                    if fit['raw_popt'] is not None:
                        # Fitter uses a screened subset with its own bin edges.
                        # Show those exact counts/edges so model height and
                        # histogram height are compared on the same basis.
                        edges = np.asarray(fit['fit_histogram_edges'])
                        fit_counts = np.asarray(fit['fit_histogram_counts'])
                        if fit['fit_polarity'] < 0:
                            edges, fit_counts = -edges[::-1], fit_counts[::-1]
                        a[0].stairs(fit_counts, edges, color='#2b6b95', lw=.9, label='Subset used for fit')
                        lo, hi = float(df.peak_integral.min()), float(df.peak_integral.max())
                        x = np.linspace(lo, hi, 1400)
                        sign = fit['fit_polarity']
                        p = np.asarray(fit['raw_popt'])
                        a[0].plot(x, stable_model(sign*x, *p), color='#b13737', lw=1.1, label='Finger fit (screened subset)')
                        for n, height in enumerate(p[7:]):
                            if height >= .01*len(df)/args.bins:
                                width = np.sqrt(p[3]**2+n*p[5]**2)
                                a[0].plot(x, height*np.exp(-.5*((sign*x-p[2]-n*p[4])/width)**2), ls='--', lw=.65, alpha=.65)
                    if fit['gain'] is not None:
                        note = f"G = {fit['gain']:.4g} ± {fit['gain_error']:.2g}\nχ²/ndf = {fit['reduced_chi2']:.3g}"
                    else:
                        note = textwrap.fill(fit['status'].replace('_', ' '), 24)
                    a[0].text(.98, .98, note, transform=a[0].transAxes, ha='right', va='top', fontsize=7,
                              bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none'})
                    a[0].legend(fontsize=6, loc='upper left')
                    for col, (x, y) in enumerate(pairs, 3):
                        hb = a[col].hexbin(df[x], df[y], gridsize=45, mincnt=1, cmap='viridis', rasterized=True)
                        fig.colorbar(hb, ax=a[col], label='Events')
                        a[col].set(xlabel=labels[x], ylabel=labels[y], title=f'{ch}: {x} vs {y}')
                    for col in range(args.examples):
                        ax = a[6+col]
                        samples = list(examples[ch].values())
                        if col < len(samples):
                            plot_pulse(ax, *samples[col], windows)
                        else:
                            ax.text(.5, .5, 'No further entries', transform=ax.transAxes, ha='center')
                            ax.set_axis_off()
                    for ax in a:
                        ax.grid(alpha=.15)
                        ax.tick_params(labelsize=7)
                        ax.xaxis.label.set_size(8)
                        ax.yaxis.label.set_size(8)
                        ax.title.set_fontsize(8)
                    summary = {'source_file': metadata['source']['path'], 'campaign': metadata['campaign'],
                               'run': metadata['run'], 'channel': ch, 'n_events': len(df),
                               'gain_adc_samples': fit['gain'], 'gain_error_adc_samples': fit['gain_error'],
                               'fit_status': fit['status'], 'fit_reason': '; '.join(fit['reasons']),
                               'fit_polarity': fit['fit_polarity'], 'reduced_chi2': fit['reduced_chi2'], 'n_fit_events': fit['n_fit_events'],
                               'baseline_mean_adc': df.baseline.mean(), 'baseline_median_adc': df.baseline.median(),
                               'baseline_spread_adc': df.baseline.std(ddof=0), 'noise_mean_adc': df.std_dev.mean(),
                               'noise_median_adc': df.std_dev.median(), 'noise_variance_mean_adc2': df.noise_variance.mean(),
                               'area_mean_adc_samples': df.peak_integral.mean(), 'area_median_adc_samples': df.peak_integral.median(),
                               'sample_period_ns': args.sample_period_ns, 'dark_count_rate_hz': None,
                               'saturation_checked': metadata['saturation_checked']}
                    if args.sample_period_ns is None and 'sample_period_ns' in df:
                        periods = df.sample_period_ns.dropna().unique()
                        if len(periods) == 1:
                            summary['sample_period_ns'] = float(periods[0])
                    for flag in ('low_snr', 'noise_zero', 'saturated', 'peak_at_search_edge'):
                        summary[flag+'_fraction'] = df[flag].mean()
                    summary['missing_crossing_fraction'] = (~df.left_crossing_found | ~df.right_crossing_found).mean()
                    for key in ('temp_c', 'bias_voltage', 'led_voltage', 'frequency_hz', 'time_ns'):
                        summary[key] = metadata['conditions'].get(key)
                    summaries.append(summary)
                    print(f"  {ch}: {fit['status']}; baseline={summary['baseline_mean_adc']:.4g} ADC, noise={summary['noise_mean_adc']:.4g} ADC", flush=True)
                bs, be, ps, pe = windows
                cond = metadata['conditions']
                condition_text = ' | '.join(f'{k}={cond[k]}' for k in ('temp_c', 'bias_voltage', 'led_voltage', 'frequency_hz') if k in cond)
                fig.suptitle(f"{metadata['campaign']} / {metadata['run']} / {Path(metadata['source']['path']).name}\n{condition_text}\nBaseline [{bs},{be}); peak search [{ps},{pe}); integration = nearest baseline crossings (inclusive)", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, .91 if len(group) == 1 else .945))
                fig.text(.995, .004, f'Page {offset//per_page+1} | area/gain: ADC-samples; noise: ADC', ha='right', fontsize=7)
                pdf.savefig(fig, dpi=140)
            finally:
                plt.close(fig)
    return fits, pd.DataFrame(summaries)


def write_final_parquet(stage, destination, fits, metadata):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    try:
        for batch in pq.ParquetFile(stage).iter_batches(batch_size=65536):
            table = pa.Table.from_batches([batch])
            channels = table['channel'].to_pylist()
            table = table.append_column('gain', pa.array([fits[c]['gain'] for c in channels], type=pa.float64()))
            table = table.append_column('gain_error', pa.array([fits[c]['gain_error'] for c in channels], type=pa.float64()))
            table = table.append_column('fit_status', pa.array([fits[c]['status'] for c in channels]))
            table = table.append_column('fit_polarity', pa.array([fits[c]['fit_polarity'] for c in channels], type=pa.int8()))
            values = table['peak_integral'].to_numpy()
            sign = np.array([fits[c]['fit_polarity'] if fits[c]['fit_polarity'] is not None else np.nan for c in channels])
            table = table.append_column('area_for_fit', pa.array(values*sign, from_pandas=True))
            schema_meta = {**(table.schema.metadata or {}), b'alpine_processing': json.dumps(json_safe(metadata), allow_nan=False).encode()}
            table = table.replace_schema_metadata(schema_meta)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def process_file(root, args, nbk, windows):
    paths = output_paths(root, args)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    settings = {'device': args.device, 'windows': list(windows), 'examples': args.examples, 'seed': args.seed,
                'channels_per_page': args.channels_per_page, 'bins': args.bins, 'num_peaks': args.num_peaks,
                'max_reduced_chi2': args.max_reduced_chi2, 'polarity': args.polarity, 'sample_period_ns': args.sample_period_ns,
                'adc_min': args.adc_min, 'adc_max': args.adc_max, 'processor_version': VERSION,
                'processor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'helper_sha256': hashlib.sha256(Path(nbk.__file__).read_bytes()).hexdigest()}
    source = source_signature(root)
    lock_path = paths['parquet'].with_suffix('.processing.lock')
    with processing_lock(lock_path):
        existing = [p for p in paths.values() if p.exists()]
        if existing and not args.overwrite:
            valid = False
            if args.skip_existing and len(existing) == len(paths):
                try:
                    m = json.loads(paths['manifest'].read_text())
                    valid = m['source'] == source and m['settings'] == settings and m['complete'] is True
                    valid &= set(m['outputs']) == {'parquet', 'pdf', 'summary'}
                    valid &= all(str(paths[k]) == info['path'] and paths[k].stat().st_size == info['size_bytes']
                                 for k, info in m['outputs'].items())
                except (OSError, ValueError, KeyError):
                    valid = False
            if valid:
                print(f'SKIPPED complete matching outputs: {root}', flush=True)
                return 'skipped'
            raise FileExistsError(f'Existing/incomplete/different outputs for {root}. Use --overwrite to replace them. First existing: {existing[0]}')
        raw, conditions = normalized_metadata(root, nbk)
        metadata = {'format_version': 2, 'source': source, 'campaign': campaign_name(root), 'run': root.parent.name,
                    'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
                    'conditions': conditions, 'helper_conditions_unmodified': raw,
                    'integration_method': 'nearest baseline crossings around strongest absolute search-window peak; inclusive signed sum',
                    'noise_definition': 'population baseline standard deviation (ddof=0)',
                    'units': {'baseline':'ADC', 'std_dev':'ADC', 'noise_variance':'ADC^2', 'peak_integral':'ADC-samples', 'gain':'ADC-samples'},
                    'saturation_checked': args.adc_min is not None or args.adc_max is not None,
                    'dark_count_rate_hz': None,
                    'dark_count_note': 'Not estimated: this selects one pulse in a trigger search window; live time and unbiased off-trigger pulse counting are required.'}
        with tempfile.TemporaryDirectory(prefix='.alpine-', dir=paths['parquet'].parent) as pt, tempfile.TemporaryDirectory(prefix='.alpine-', dir=paths['pdf'].parent) as vt:
            feature_stage = Path(pt) / 'features.parquet'
            channels, examples, counts, scalars = extract_features(root, feature_stage, nbk, args, windows)
            metadata.update(channels=channels, entries_per_channel=counts, preserved_scalar_branches=scalars,
                            example_entries={ch: list(examples[ch]) for ch in channels})
            staged = {'parquet':Path(pt)/'final.parquet', 'pdf':Path(vt)/'report.pdf',
                      'summary':Path(vt)/'summary.csv', 'manifest':Path(vt)/'manifest.json'}
            fits, summary = make_report(feature_stage, staged['pdf'], channels, examples, nbk, args, windows, metadata)
            metadata['fits'] = fits
            write_final_parquet(feature_stage, staged['parquet'], fits, metadata)
            summary.to_csv(staged['summary'], index=False)
            if source_signature(root) != source:
                raise RuntimeError('Source changed during processing; outputs not published')
            metadata['outputs'] = {k: {'path':str(paths[k]), 'size_bytes':staged[k].stat().st_size} for k in ('parquet','pdf','summary')}
            metadata['complete'] = True
            staged['manifest'].write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False)+'\n')
            for path in staged.values():
                path.chmod(0o664)
            # The completion manifest is published last. Remove old marker first
            # so an interrupted overwrite cannot masquerade as a complete pair.
            paths['manifest'].unlink(missing_ok=True)
            for key in ('parquet', 'pdf', 'summary', 'manifest'):
                os.replace(staged[key], paths[key])
        print(f"SAVED {sum(counts.values())} channel-events: {root}", flush=True)
        for key, path in paths.items():
            print(f'  {key}: {path}', flush=True)
    return 'saved'


def collect_inputs(inputs):
    found = set()
    for item in inputs:
        item = item.expanduser().resolve(strict=True)
        if item.is_dir():
            found.update(p.resolve() for p in item.rglob('*.root') if p.is_file())
        elif item.is_file() and item.suffix.lower() == '.root':
            found.add(item)
        else:
            raise ValueError(f'Not a ROOT file or folder: {item}')
    if not found:
        raise ValueError('No ROOT files found')
    return sorted(found)


def main():
    args = parse_args()
    files = collect_inputs(args.inputs)
    nbk = load_helper()
    windows = windows_from_helper(args, nbk)
    print(f'ALPINE processor {VERSION}: {len(files)} ROOT file(s), device={args.device}, windows={windows}', flush=True)
    counts = {'saved':0, 'skipped':0, 'failed':0}
    for root in files:
        print(f'\nProcessing: {root}', flush=True)
        try:
            counts[process_file(root, args, nbk, windows)] += 1
        except Exception as e:
            counts['failed'] += 1
            print(f'FAILED {root}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
    print(f"\nBatch finished: {counts['saved']} saved, {counts['skipped']} skipped, {counts['failed']} failed", flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, ImportError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
