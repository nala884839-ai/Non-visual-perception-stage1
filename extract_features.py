#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_features.py — Final Stage 2 redesigned feature extractor
=================================================================
Raw trial folders -> redesigned strict-modality feature table CSV.

This is the final Stage 2 feature-extraction version used for the
redesigned ML results.

Key design choices
------------------
1. Strict modality separation
   - T  : tactile-only features
   - A  : acoustic-only features
   - O  : olfactory-only features
   - TA : tactile-acoustic cross features only
   - TO : tactile-olfactory cross features only
   - AO : acoustic-olfactory cross features only
   This prevents acoustic-derived cross features from entering T+O.

2. Acoustic redesign
   - Burst-wise DC removal before spectral/energy features.
   - Tap-contact aligned burst selection.
   - Tap-wise energy, RMS, peak, SNR, centroid, dominant frequency,
     band ratios, high-frequency ratio, and CV features.

3. Olfactory redesign
   - BME688 absolute gas resistance is kept only as QC.
   - Main BME688 inputs are baseline-normalized heater-profile responses:
       O_BME_dlogR_Txxx = log(R_baseline_Txxx) - log(R_sniff_Txxx)
     which reduces direct session/baseline leakage.

4. ML-ready output option
   - Drops high-missingness columns used only in the full diagnostic table.

Usage examples
--------------
    python extract_features.py --data-root data/raw/stage2a --stage 2a \
        --out feature_table_stage2a_final.csv --ml-ready

    python extract_features.py --data-root data/raw/stage2b --stage 2b \
        --out feature_table_stage2b_final.csv --groups-out feature_groups_stage2b.json

Expected folder layout
----------------------
    <data-root>/obj01_orange/trial_001/*.csv
    <data-root>/obj02_apple/trial_001/*.csv

or any tree where trial folders match --glob (default: **/trial_*).
"""
from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

FS_ACO = 5000.0
BME_TEMPS = [100, 133, 167, 200, 233, 267, 300, 333, 367, 400]

# ---------- utilities ----------

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    try:
        return float(np.polyfit(x[m], y[m], 1)[0])
    except Exception:
        return np.nan


def auc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.trapezoid(y[m], x[m]))


def cv(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    mu = np.mean(x)
    return float(np.std(x) / (abs(mu) + 1e-12))


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * mad + 1e-9)


def phase_intervals(events: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """Return phase intervals in microseconds from event file.
    Handles *_start/*_end pairs and repeated names."""
    seg: Dict[str, List[Tuple[float, float]]] = {}
    if events is None or len(events) == 0:
        return seg
    tcol = 't_us' if 't_us' in events.columns else events.columns[0]
    pcol = 'phase_name' if 'phase_name' in events.columns else events.columns[-1]
    rows = [(safe_float(r[tcol]), str(r[pcol]).strip()) for _, r in events.iterrows()]
    starts: Dict[str, float] = {}
    for t, name in rows:
        if not math.isfinite(t):
            continue
        if name.endswith('_start'):
            base = name[:-6]
            starts[base] = t
        elif name.endswith('_end'):
            base = name[:-4]
            if base in starts:
                seg.setdefault(base, []).append((starts[base], t))
                del starts[base]
    # If tap only has tap_start/tap_end and tap_phase_start/end also exist, keep both.
    # Add contact/release subintervals as contact windows.
    contact_starts = []
    for t, name in rows:
        m = re.match(r'tap_(\d+)_contact', name)
        if m:
            contact_starts.append((m.group(1), t))
        m2 = re.match(r'tap_(\d+)_release', name)
        if m2:
            for idx, t0 in contact_starts:
                if idx == m2.group(1):
                    seg.setdefault(f'tap{idx}_contact', []).append((t0, t))
                    break
    return seg


def event_time(events: pd.DataFrame, name: str) -> float:
    if events is None: return np.nan
    pcol = 'phase_name' if 'phase_name' in events.columns else events.columns[-1]
    tcol = 't_us' if 't_us' in events.columns else events.columns[0]
    m = events[pcol].astype(str).str.strip() == name
    if not m.any(): return np.nan
    return safe_float(events.loc[m, tcol].iloc[0])


def mask_phase(t_us, seg, phase: str):
    t_us = np.asarray(t_us, dtype=float)
    mask = np.zeros(len(t_us), dtype=bool)
    for t0, t1 in seg.get(phase, []):
        mask |= (t_us >= t0) & (t_us < t1)
    return mask

# ---------- tactile ----------

def tactile_features(trial: Path, seg: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
    out = {}
    d = read_csv(trial / 'tactile.csv')
    if d is None or len(d) < 5:
        return {k: np.nan for k in T_KEYS}
    t_us = d['t_us'].to_numpy(float)
    t = t_us / 1e6
    # Keep both total and s2+s3 because s1 is often weak/zero but still diagnostic.
    s1 = d.get('s1', pd.Series(np.zeros(len(d)))).to_numpy(float)
    s2 = d.get('s2', pd.Series(np.zeros(len(d)))).to_numpy(float)
    s3 = d.get('s3', pd.Series(np.zeros(len(d)))).to_numpy(float)
    total_raw = s1 + s2 + s3
    valid23_raw = s2 + s3
    base_m = mask_phase(t_us, seg, 'baseline')
    tap_m = mask_phase(t_us, seg, 'tap_phase')
    if tap_m.sum() < 3:
        tap_m = mask_phase(t_us, seg, 'tap')
    if tap_m.sum() < 3:
        tap_m = np.ones(len(d), dtype=bool)
    base = np.nanmedian(valid23_raw[base_m]) if base_m.sum() >= 3 else np.nanmedian(valid23_raw[:max(3, len(d)//20)])
    y = valid23_raw - base
    yt = y[tap_m]
    tt = t[tap_m]
    if len(yt) < 3:
        return {k: np.nan for k in T_KEYS}
    peak = float(np.nanmax(yt))
    ipk_local = int(np.nanargmax(yt))
    ipk_global = np.where(tap_m)[0][ipk_local]
    thr = max(0.2*peak, np.nanmedian(yt)+3*np.nanstd(y[base_m]) if base_m.sum()>=5 else 0.2*peak)
    contact = yt > thr
    # contact segments and peaks by threshold crossing
    seg_peaks = []
    seg_durs = []
    if contact.any():
        idx = np.where(contact)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for g in groups:
            if len(g) >= 1:
                seg_peaks.append(float(np.nanmax(yt[g])))
                seg_durs.append(float(tt[g[-1]] - tt[g[0]]))
    contact_time = float(np.sum(contact) / 100.0)  # tactile nominal 100 Hz
    # Use actual first-last contact too
    contact_span = float(tt[contact][-1] - tt[contact][0]) if contact.any() else 0.0
    out.update({
        'T_peak_sum23': peak,
        'T_peak_total123': float(np.nanmax(total_raw[tap_m] - (np.nanmedian(total_raw[base_m]) if base_m.sum()>=3 else np.nanmedian(total_raw[:max(3, len(d)//20)])))),
        'T_time_to_peak_s': float(tt[ipk_local] - tt[0]),
        'T_rise_slope': slope(tt[:ipk_local+1], yt[:ipk_local+1]) if ipk_local >= 1 else np.nan,
        'T_impulse_pos': auc(tt, np.clip(yt, 0, None)),
        'T_contact_duration_count_s': contact_time,
        'T_contact_span_s': contact_span,
        'T_contact_event_count': float(len(seg_peaks)),
        'T_tap_peak_cv': cv(seg_peaks),
        'T_tap_duration_cv': cv(seg_durs),
        'T_steady_mean': float(np.nanmean(yt[contact])) if contact.any() else np.nan,
        'T_steady_var': float(np.nanvar(yt[contact])) if contact.any() else np.nan,
        'T_baseline_noise_std': float(np.nanstd(y[base_m])) if base_m.sum() >= 3 else np.nan,
        'T_residual_rms': float(np.sqrt(np.nanmean((yt - np.nanmedian(yt))**2))),
        'T_peak_s2': float(np.nanmax(s2[tap_m] - (np.nanmedian(s2[base_m]) if base_m.sum()>=3 else np.nanmedian(s2[:max(3,len(d)//20)])))),
        'T_peak_s3': float(np.nanmax(s3[tap_m] - (np.nanmedian(s3[base_m]) if base_m.sum()>=3 else np.nanmedian(s3[:max(3,len(d)//20)])))),
    })
    out['T_s2_s3_peak_ratio'] = out['T_peak_s2'] / (out['T_peak_s3'] + 1e-9)
    # centroid during contact if available
    if 'cx' in d.columns and 'cy' in d.columns and contact.any():
        tap_idx = np.where(tap_m)[0]
        cidx = tap_idx[contact]
        cx = d['cx'].to_numpy(float)[cidx]
        cy = d['cy'].to_numpy(float)[cidx]
        cx = cx[cx >= 0]
        cy = cy[cy >= 0]
        out['T_contact_cx_mean'] = float(np.nanmean(cx)) if len(cx) else np.nan
        out['T_contact_cy_mean'] = float(np.nanmean(cy)) if len(cy) else np.nan
        out['T_contact_centroid_var'] = float(np.nanvar(cx) + np.nanvar(cy)) if len(cx) and len(cy) else np.nan
    else:
        out['T_contact_cx_mean'] = np.nan
        out['T_contact_cy_mean'] = np.nan
        out['T_contact_centroid_var'] = np.nan
    # release decay after peak on positive envelope
    post_t = tt[ipk_local:]
    post_y = np.clip(yt[ipk_local:], 1e-9, None)
    if len(post_y) >= 5 and np.nanmax(post_y) > 0:
        # fit only until signal falls near threshold or 0.5 s
        m = (post_t - post_t[0]) <= 0.5
        try:
            b = np.polyfit(post_t[m] - post_t[m][0], np.log(post_y[m]), 1)[0]
            out['T_release_tau_s'] = float(-1.0/b) if b < 0 else np.nan
        except Exception:
            out['T_release_tau_s'] = np.nan
    else:
        out['T_release_tau_s'] = np.nan
    return {k: out.get(k, np.nan) for k in T_KEYS}

T_KEYS = [
    'T_peak_sum23','T_peak_total123','T_time_to_peak_s','T_rise_slope','T_impulse_pos',
    'T_contact_duration_count_s','T_contact_span_s','T_contact_event_count','T_tap_peak_cv','T_tap_duration_cv',
    'T_steady_mean','T_steady_var','T_baseline_noise_std','T_residual_rms','T_peak_s2','T_peak_s3',
    'T_s2_s3_peak_ratio','T_contact_cx_mean','T_contact_cy_mean','T_contact_centroid_var','T_release_tau_s'
]

# ---------- acoustic ----------

def load_acoustic_bursts(path: Path):
    bursts = []
    if not path.exists():
        return bursts
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) < 3:
                continue
            try:
                t0 = float(parts[0])
                vals = np.array([float(x) for x in parts[2].split()], dtype=float)
            except Exception:
                continue
            if len(vals) >= 16:
                bursts.append((t0, vals))
    return bursts


def spectral_features(sig: np.ndarray):
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.nanmean(sig)
    if len(sig) < 16:
        return dict(energy=np.nan, rms=np.nan, peak=np.nan, p2p=np.nan, zcr=np.nan,
                    centroid=np.nan, flatness=np.nan, dom=np.nan, rolloff=np.nan,
                    low=np.nan, mid=np.nan, high=np.nan, high_ratio=np.nan)
    # Hann taper reduces impact of edge discontinuities.
    w = sig * np.hanning(len(sig))
    energy = float(np.sum(w**2))
    rms = float(np.sqrt(np.mean(w**2)))
    peak = float(np.max(np.abs(w)))
    p2p = float(np.ptp(sig))
    zcr = float(np.mean(np.diff(np.signbit(w)) != 0))
    freqs = np.fft.rfftfreq(len(w), 1.0 / FS_ACO)
    P = np.abs(np.fft.rfft(w))**2
    # Drop DC and <100Hz for acoustic material features.
    mask = freqs >= 100
    if mask.sum() == 0 or np.sum(P[mask]) <= 0:
        centroid=flatness=dom=rolloff=low=mid=high=high_ratio=np.nan
    else:
        f = freqs[mask]
        p = P[mask]
        psum = np.sum(p) + 1e-12
        centroid = float(np.sum(f*p)/psum)
        flatness = float(np.exp(np.mean(np.log(p + 1e-20))) / (np.mean(p) + 1e-20))
        dom = float(f[np.argmax(p)])
        c = np.cumsum(p)
        rolloff = float(f[min(np.searchsorted(c, 0.85*c[-1]), len(f)-1)])
        def bp(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(np.sum(P[m]) / (np.sum(P[(freqs>=100)]) + 1e-12))
        low = bp(100, 500)
        mid = bp(500, 1500)
        high = bp(1500, 2500)
        high_ratio = high / (low + mid + high + 1e-12)
    return dict(energy=energy, rms=rms, peak=peak, p2p=p2p, zcr=zcr,
                centroid=centroid, flatness=flatness, dom=dom, rolloff=rolloff,
                low=low, mid=mid, high=high, high_ratio=high_ratio)


def acoustic_features(trial: Path, events: pd.DataFrame, seg: Dict[str, List[Tuple[float, float]]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return A features and auxiliary per-trial values used for cross-modal."""
    bursts = load_acoustic_bursts(trial / 'acoustic.csv')
    if not bursts:
        return {k: np.nan for k in A_KEYS}, {}
    # Compute features for all bursts.
    t_b = np.array([b[0] for b in bursts], dtype=float)
    feat_list = [spectral_features(v) for _, v in bursts]
    energies = np.array([f['energy'] for f in feat_list], dtype=float)
    peaks = np.array([f['peak'] for f in feat_list], dtype=float)
    quiet_m = np.zeros(len(bursts), dtype=bool)
    q0 = event_time(events, 'tap_quiet_start')
    q1 = event_time(events, 'tap_quiet_end')
    if math.isfinite(q0) and math.isfinite(q1):
        quiet_m = (t_b >= q0) & (t_b < q1)
    # contact-aligned bursts: choose strongest burst within each tap contact interval +/-100ms.
    tap_burst_idx = []
    per_tap = []
    for i in [1,2,3]:
        intervals = seg.get(f'tap{i}_contact', [])
        if not intervals:
            continue
        t0, t1 = intervals[0]
        m = (t_b >= t0 - 100_000) & (t_b <= t1 + 150_000)
        idxs = np.where(m)[0]
        if len(idxs):
            best = idxs[np.nanargmax(energies[idxs])]
            tap_burst_idx.append(int(best))
            per_tap.append(feat_list[best])
    if not tap_burst_idx:
        # fallback: use top 3 energetic bursts during tap_phase or tap.
        tm = mask_phase(t_b, seg, 'tap_phase')
        if tm.sum() < 1:
            tm = mask_phase(t_b, seg, 'tap')
        idxs = np.where(tm)[0]
        if len(idxs) == 0:
            idxs = np.arange(len(bursts))
        top = idxs[np.argsort(energies[idxs])[-min(3,len(idxs)):]]
        tap_burst_idx = list(map(int, top))
        per_tap = [feat_list[i] for i in tap_burst_idx]
    tap_idx = np.array(tap_burst_idx, dtype=int)
    # Quiet baseline for SNR; fallback earliest low-energy bursts.
    if quiet_m.sum() < 1:
        quiet_idx = np.argsort(energies)[:min(3, len(energies))]
    else:
        quiet_idx = np.where(quiet_m)[0]
    quiet_energy = float(np.nanmedian(energies[quiet_idx])) if len(quiet_idx) else np.nan
    quiet_rms = float(np.nanmedian([feat_list[i]['rms'] for i in quiet_idx])) if len(quiet_idx) else np.nan
    def arr(name): return np.array([f[name] for f in per_tap], dtype=float)
    out = {
        'A_tap_energy_mean': float(np.nanmean(arr('energy'))),
        'A_tap_energy_max': float(np.nanmax(arr('energy'))),
        'A_tap_energy_cv': cv(arr('energy')),
        'A_tap_rms_mean': float(np.nanmean(arr('rms'))),
        'A_tap_peak_mean': float(np.nanmean(arr('peak'))),
        'A_tap_peak_max': float(np.nanmax(arr('peak'))),
        'A_tap_p2p_mean': float(np.nanmean(arr('p2p'))),
        'A_energy_snr_median': float(np.nanmedian(arr('energy'))/(quiet_energy+1e-12)) if np.isfinite(quiet_energy) else np.nan,
        'A_rms_snr_median': float(np.nanmedian(arr('rms'))/(quiet_rms+1e-12)) if np.isfinite(quiet_rms) else np.nan,
        'A_centroid_mean': float(np.nanmean(arr('centroid'))),
        'A_centroid_cv': cv(arr('centroid')),
        'A_dominant_freq_mean': float(np.nanmean(arr('dom'))),
        'A_dominant_freq_cv': cv(arr('dom')),
        'A_rolloff85_mean': float(np.nanmean(arr('rolloff'))),
        'A_flatness_mean': float(np.nanmean(arr('flatness'))),
        'A_zcr_mean': float(np.nanmean(arr('zcr'))),
        'A_band_low_mean': float(np.nanmean(arr('low'))),
        'A_band_mid_mean': float(np.nanmean(arr('mid'))),
        'A_band_high_mean': float(np.nanmean(arr('high'))),
        'A_high_ratio_mean': float(np.nanmean(arr('high_ratio'))),
        'A_high_ratio_cv': cv(arr('high_ratio')),
        'A_quiet_energy_median': quiet_energy,
        'A_n_selected_tap_bursts': float(len(tap_idx)),
    }
    # Ringing / decay on strongest burst envelope after impact.
    strongest = int(tap_idx[np.nanargmax(energies[tap_idx])]) if len(tap_idx) else int(np.nanargmax(energies))
    sig = bursts[strongest][1].astype(float) - np.mean(bursts[strongest][1])
    env = np.abs(sig)
    pk = int(np.argmax(env))
    post = env[pk:]
    if len(post) >= 8:
        nf = np.nanmedian(env[:max(8, len(env)//8)]) + 3*np.nanstd(env[:max(8, len(env)//8)])
        above = np.where(post > nf)[0]
        out['A_ringing_duration_s'] = float(above[-1]/FS_ACO) if len(above) else 0.0
        tp = np.arange(len(post)) / FS_ACO
        pclip = np.clip(post, 1e-9, None)
        try:
            b = np.polyfit(tp[:min(len(tp), int(0.08*FS_ACO))], np.log(pclip[:min(len(tp), int(0.08*FS_ACO))]), 1)[0]
            out['A_decay_tau_s'] = float(-1.0/b) if b < 0 else np.nan
        except Exception:
            out['A_decay_tau_s'] = np.nan
    else:
        out['A_ringing_duration_s'] = np.nan
        out['A_decay_tau_s'] = np.nan
    aux = {
        'tap_burst_times_us': t_b[tap_idx] if len(tap_idx) else np.array([]),
        'tap_burst_energies': energies[tap_idx] if len(tap_idx) else np.array([]),
        'tap_burst_high_ratio': arr('high_ratio') if len(per_tap) else np.array([]),
        'tap_burst_centroid': arr('centroid') if len(per_tap) else np.array([]),
        'quiet_energy': quiet_energy,
    }
    return {k: out.get(k, np.nan) for k in A_KEYS}, aux

A_KEYS = [
    'A_tap_energy_mean','A_tap_energy_max','A_tap_energy_cv','A_tap_rms_mean','A_tap_peak_mean','A_tap_peak_max',
    'A_tap_p2p_mean','A_energy_snr_median','A_rms_snr_median','A_centroid_mean','A_centroid_cv','A_dominant_freq_mean',
    'A_dominant_freq_cv','A_rolloff85_mean','A_flatness_mean','A_zcr_mean','A_band_low_mean','A_band_mid_mean',
    'A_band_high_mean','A_high_ratio_mean','A_high_ratio_cv','A_quiet_energy_median','A_n_selected_tap_bursts',
    'A_ringing_duration_s','A_decay_tau_s'
]

# ---------- olfactory ----------

def olfactory_features(trial: Path, seg: Dict[str, List[Tuple[float, float]]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    out = {}
    ens = read_csv(trial / 'gas_ens160.csv')
    if ens is not None and len(ens) >= 3:
        t_us = ens['t_us'].to_numpy(float)
        t = t_us / 1e6
        tvoc = ens['tvoc_ppb'].to_numpy(float)
        eco2 = ens['eco2_ppm'].to_numpy(float) if 'eco2_ppm' in ens.columns else np.full(len(ens), np.nan)
        mb = mask_phase(t_us, seg, 'baseline') | mask_phase(t_us, seg, 'hover')
        ms = mask_phase(t_us, seg, 'sniff')
        mr = mask_phase(t_us, seg, 'recovery')
        if mb.sum() < 2: mb = np.arange(len(ens)) < max(3, len(ens)//8)
        if ms.sum() < 2: ms = np.ones(len(ens), dtype=bool)
        base = np.nanmedian(tvoc[mb])
        dtvoc = tvoc - base
        out.update({
            'O_ENS_base_tvoc': float(base),
            'O_ENS_sniff_delta_mean': float(np.nanmean(dtvoc[ms])),
            'O_ENS_sniff_delta_max': float(np.nanmax(dtvoc[ms])),
            'O_ENS_sniff_slope': slope(t[ms], dtvoc[ms]),
            'O_ENS_sniff_auc': auc(t[ms], dtvoc[ms]),
            'O_ENS_recovery_delta_mean': float(np.nanmean(dtvoc[mr])) if mr.sum()>=2 else np.nan,
            'O_ENS_eco2_delta_mean': float(np.nanmean(eco2[ms] - np.nanmedian(eco2[mb]))) if np.isfinite(eco2).any() else np.nan,
        })
    else:
        for k in ENS_KEYS: out[k] = np.nan

    bme = read_csv(trial / 'gas_bme688.csv')
    dlog_by_temp = {T: np.nan for T in BME_TEMPS}
    if bme is not None and len(bme) >= 10 and {'t_us','set_temp_c','gas_resistance_ohm'}.issubset(bme.columns):
        t_us = bme['t_us'].to_numpy(float)
        t = t_us / 1e6
        temps = bme['set_temp_c'].to_numpy(float)
        R = bme['gas_resistance_ohm'].to_numpy(float)
        valid = bme['gas_valid'].to_numpy(float) > 0 if 'gas_valid' in bme.columns else np.ones(len(bme), dtype=bool)
        mb = (mask_phase(t_us, seg, 'baseline') | mask_phase(t_us, seg, 'hover')) & valid
        ms = mask_phase(t_us, seg, 'sniff') & valid
        mr = mask_phase(t_us, seg, 'recovery') & valid
        if mb.sum() < 5:
            mb = valid & (np.arange(len(bme)) < max(10, len(bme)//8))
        if ms.sum() < 5:
            ms = valid
        base_R = []
        sniff_R = []
        rec_R = []
        for T in BME_TEMPS:
            mt = np.abs(temps - T) <= 1.0
            rb = np.nanmedian(R[mb & mt]) if (mb & mt).sum() >= 1 else np.nan
            rs = np.nanmedian(R[ms & mt]) if (ms & mt).sum() >= 1 else np.nan
            rr = np.nanmedian(R[mr & mt]) if (mr & mt).sum() >= 1 else np.nan
            base_R.append(rb); sniff_R.append(rs); rec_R.append(rr)
            # positive if sniff resistance lower than baseline (VOC response)
            if np.isfinite(rb) and np.isfinite(rs) and rb > 0 and rs > 0:
                dlog_by_temp[T] = float(np.log(rb) - np.log(rs))
            out[f'O_BME_dlogR_T{T}'] = dlog_by_temp[T]
        base_R = np.array(base_R, dtype=float)
        sniff_R = np.array(sniff_R, dtype=float)
        dlog = np.array([dlog_by_temp[T] for T in BME_TEMPS], dtype=float)
        tarr = np.array(BME_TEMPS, dtype=float)
        finite = np.isfinite(dlog)
        if finite.sum() >= 3:
            out.update({
                'O_BME_dlogR_mean': float(np.nanmean(dlog)),
                'O_BME_dlogR_std': float(np.nanstd(dlog)),
                'O_BME_dlogR_max': float(np.nanmax(dlog)),
                'O_BME_dlogR_min': float(np.nanmin(dlog)),
                'O_BME_dlogR_range': float(np.nanmax(dlog)-np.nanmin(dlog)),
                'O_BME_dlogR_abs_auc_temp': auc(tarr[finite], np.abs(dlog[finite])),
                'O_BME_dlogR_signed_auc_temp': auc(tarr[finite], dlog[finite]),
                'O_BME_dlogR_slope_temp': slope(tarr[finite], dlog[finite]),
                'O_BME_temp_at_max_dlogR': float(tarr[finite][np.nanargmax(dlog[finite])]),
                'O_BME_low_high_dlogR_ratio': float(np.nanmean(dlog[:3])/(np.nanmean(dlog[-3:])+1e-12)),
                'O_BME_profile_curvature': float(np.polyfit(tarr[finite], dlog[finite], 2)[0]) if finite.sum() >= 4 else np.nan,
                'O_BME_sniff_logR_slope_temp': slope(tarr[np.isfinite(sniff_R)&(sniff_R>0)], np.log(sniff_R[np.isfinite(sniff_R)&(sniff_R>0)])),
                'O_BME_base_logR_slope_temp': slope(tarr[np.isfinite(base_R)&(base_R>0)], np.log(base_R[np.isfinite(base_R)&(base_R>0)])),
            })
        else:
            for k in BME_SUMMARY_KEYS: out[k] = np.nan
        # absolute features as QC only—not meant as main O class cue; include with QC_ prefix.
        out['QC_BME_base_logR_mean'] = float(np.nanmean(np.log(base_R[np.isfinite(base_R)&(base_R>0)]))) if np.any(np.isfinite(base_R)&(base_R>0)) else np.nan
        out['QC_BME_sniff_logR_mean'] = float(np.nanmean(np.log(sniff_R[np.isfinite(sniff_R)&(sniff_R>0)]))) if np.any(np.isfinite(sniff_R)&(sniff_R>0)) else np.nan
        out['QC_BME_valid_temp_count'] = float(finite.sum())
    else:
        for k in BME_TEMP_KEYS + BME_SUMMARY_KEYS + ['QC_BME_base_logR_mean','QC_BME_sniff_logR_mean','QC_BME_valid_temp_count']:
            out[k] = np.nan
    aux = {
        'bme_dlog_mean': out.get('O_BME_dlogR_mean', np.nan),
        'bme_dlog_max': out.get('O_BME_dlogR_max', np.nan),
        'bme_dlog_high': np.nanmean([out.get(f'O_BME_dlogR_T{T}', np.nan) for T in [300,333,367,400]]),
        'ens_delta': out.get('O_ENS_sniff_delta_mean', np.nan),
    }
    return {k: out.get(k, np.nan) for k in O_KEYS + QC_O_KEYS}, aux

ENS_KEYS = ['O_ENS_base_tvoc','O_ENS_sniff_delta_mean','O_ENS_sniff_delta_max','O_ENS_sniff_slope','O_ENS_sniff_auc','O_ENS_recovery_delta_mean','O_ENS_eco2_delta_mean']
BME_TEMP_KEYS = [f'O_BME_dlogR_T{T}' for T in BME_TEMPS]
BME_SUMMARY_KEYS = ['O_BME_dlogR_mean','O_BME_dlogR_std','O_BME_dlogR_max','O_BME_dlogR_min','O_BME_dlogR_range','O_BME_dlogR_abs_auc_temp','O_BME_dlogR_signed_auc_temp','O_BME_dlogR_slope_temp','O_BME_temp_at_max_dlogR','O_BME_low_high_dlogR_ratio','O_BME_profile_curvature','O_BME_sniff_logR_slope_temp','O_BME_base_logR_slope_temp']
O_KEYS = ENS_KEYS + BME_TEMP_KEYS + BME_SUMMARY_KEYS
QC_O_KEYS = ['QC_BME_base_logR_mean','QC_BME_sniff_logR_mean','QC_BME_valid_temp_count']

# ---------- cross-modal strict pairwise ----------

def cross_features(trial: Path, events: pd.DataFrame, seg, T: Dict[str,float], A: Dict[str,float], O: Dict[str,float], a_aux: Dict, o_aux: Dict) -> Dict[str, float]:
    out = {}
    # TA only: all acoustic-derived cross-modal features must be in TA, not TO.
    out['TA_energy_per_force'] = A.get('A_tap_energy_mean', np.nan)/(T.get('T_peak_sum23', np.nan)+1e-12)
    out['TA_peakamp_per_force'] = A.get('A_tap_peak_mean', np.nan)/(T.get('T_peak_sum23', np.nan)+1e-12)
    out['TA_highratio_times_tactile_var'] = A.get('A_high_ratio_mean', np.nan)*T.get('T_steady_var', np.nan)
    out['TA_force_energy_ratio'] = T.get('T_peak_sum23', np.nan)/(A.get('A_tap_energy_mean', np.nan)+1e-12)
    # lag / corr between tactile contact events and acoustic burst energy.
    td = read_csv(trial / 'tactile.csv')
    if td is not None and len(td) >= 5 and len(a_aux.get('tap_burst_times_us', [])) >= 1:
        t_us = td['t_us'].to_numpy(float)
        force = td['s2'].to_numpy(float) + td['s3'].to_numpy(float)
        tap_m = mask_phase(t_us, seg, 'tap_phase')
        if tap_m.sum() < 3: tap_m = mask_phase(t_us, seg, 'tap')
        if tap_m.sum() < 3: tap_m = np.ones(len(td), dtype=bool)
        ttp = t_us[tap_m]
        ffp = force[tap_m]
        t_force_peak = ttp[int(np.argmax(ffp))]
        b_t = np.asarray(a_aux.get('tap_burst_times_us', []), dtype=float)
        b_e = np.asarray(a_aux.get('tap_burst_energies', []), dtype=float)
        if len(b_t) and len(b_e):
            out['TA_lag_acoustic_minus_tactile_s'] = float((b_t[int(np.nanargmax(b_e))] - t_force_peak)/1e6)
        if len(b_t) >= 3 and np.std(b_e)>0:
            f_at_b = np.interp(b_t, ttp, ffp)
            out['TA_force_acoustic_corr'] = float(np.corrcoef(f_at_b, b_e)[0,1]) if np.std(f_at_b)>0 else np.nan
        else:
            out['TA_force_acoustic_corr'] = np.nan
    else:
        out['TA_lag_acoustic_minus_tactile_s'] = np.nan
        out['TA_force_acoustic_corr'] = np.nan
    # TO: tactile + olfactory only. No acoustic-derived values.
    out['TO_force_times_bme_dlog_mean'] = T.get('T_peak_sum23', np.nan)*o_aux.get('bme_dlog_mean', np.nan)
    out['TO_contactduration_times_bme_dlog_mean'] = T.get('T_contact_duration_count_s', np.nan)*o_aux.get('bme_dlog_mean', np.nan)
    out['TO_tactilevar_times_bme_dlog_range'] = T.get('T_steady_var', np.nan)*O.get('O_BME_dlogR_range', np.nan)
    out['TO_force_times_ens_delta'] = T.get('T_peak_sum23', np.nan)*o_aux.get('ens_delta', np.nan)
    out['TO_tactile_gas_balance'] = T.get('T_peak_sum23', np.nan)/(abs(o_aux.get('bme_dlog_mean', np.nan))+1e-12)
    # AO: acoustic + olfactory only. No tactile-derived values.
    out['AO_energy_times_bme_dlog_mean'] = A.get('A_tap_energy_mean', np.nan)*o_aux.get('bme_dlog_mean', np.nan)
    out['AO_highratio_times_bme_high_dlog'] = A.get('A_high_ratio_mean', np.nan)*o_aux.get('bme_dlog_high', np.nan)
    out['AO_centroid_times_bme_dlog_slope'] = A.get('A_centroid_mean', np.nan)*O.get('O_BME_dlogR_slope_temp', np.nan)
    out['AO_energy_times_ens_delta'] = A.get('A_tap_energy_mean', np.nan)*o_aux.get('ens_delta', np.nan)
    out['AO_acoustic_gas_balance'] = A.get('A_tap_energy_mean', np.nan)/(abs(o_aux.get('bme_dlog_mean', np.nan))+1e-12)
    return {k: out.get(k, np.nan) for k in X_KEYS}

TA_KEYS = ['TA_energy_per_force','TA_peakamp_per_force','TA_highratio_times_tactile_var','TA_force_energy_ratio','TA_lag_acoustic_minus_tactile_s','TA_force_acoustic_corr']
TO_KEYS = ['TO_force_times_bme_dlog_mean','TO_contactduration_times_bme_dlog_mean','TO_tactilevar_times_bme_dlog_range','TO_force_times_ens_delta','TO_tactile_gas_balance']
AO_KEYS = ['AO_energy_times_bme_dlog_mean','AO_highratio_times_bme_high_dlog','AO_centroid_times_bme_dlog_slope','AO_energy_times_ens_delta','AO_acoustic_gas_balance']
X_KEYS = TA_KEYS + TO_KEYS + AO_KEYS

# ---------- dataset ----------

def label_from_folder(folder: str, stage: str):
    name = folder.split('_', 1)[1] if '_' in folder else folder
    if stage == 'stage1':
        # Stage 1: 3 materials × 3 contents = 9 classes
        order = [
            'glass_empty', 'glass_ethanol', 'glass_acetone',
            'ceramic_empty', 'ceramic_ethanol', 'ceramic_acetone',
            'plastic_empty', 'plastic_ethanol', 'plastic_acetone',
        ]
    elif stage == 'stage2a':
        # Stage 2A: fruit / ball 8-class task
        order = ['orange', 'apple', 'pear', 'grapefruit', 'pingpong', 'golf', 'tennis', 'baseball']
    elif stage == 'stage2b':
        # Stage 2B: skin-like 3-class task
        order = ['skin', 'dragonskin', 'chicken']
    else:
        order = []
    try:
        return name, order.index(name)
    except ValueError:
        return name, np.nan


def extract_trial(trial: Path, stage: str) -> Dict[str, float]:
    events = read_csv(trial / 'events.csv')
    seg = phase_intervals(events)
    meta = {}
    if (trial / 'meta.json').exists():
        try:
            meta = json.loads((trial / 'meta.json').read_text(encoding='utf-8'))
        except Exception:
            meta = {}
    obj_folder = trial.parent.name
    obj_name, label = label_from_folder(obj_folder, stage)
    row = {
        'dataset': stage,
        'session': meta.get('session'),
        'obj_id': meta.get('obj_id'),
        'obj_name': meta.get('obj_name', obj_name),
        'label': meta.get('label', label),
        'group': meta.get('group', obj_name),
        'trial_id': meta.get('trial_id', re.sub(r'\D','',trial.name)[:3]),
        'variant': meta.get('variant', 'tap'),
        'trial_dir': str(trial),
    }
    # QC counts
    for name in ['tactile','gas_ens160','gas_bme688','environment']:
        df = read_csv(trial / f'{name}.csv')
        row[f'QC_n_{name}'] = len(df) if df is not None else 0
    row['QC_n_acoustic_bursts'] = len(load_acoustic_bursts(trial / 'acoustic.csv'))
    T = tactile_features(trial, seg)
    A, a_aux = acoustic_features(trial, events, seg)
    O, o_aux = olfactory_features(trial, seg)
    X = cross_features(trial, events, seg, T, A, O, a_aux, o_aux)
    row.update(T); row.update(A); row.update(O); row.update(X)
    # Useful QC flags
    row['QC_ok_tactile'] = int(row['QC_n_tactile'] > 8000 and np.isfinite(row.get('T_peak_sum23', np.nan)))
    row['QC_ok_acoustic'] = int(row['QC_n_acoustic_bursts'] >= 3 and np.isfinite(row.get('A_tap_energy_mean', np.nan)))
    row['QC_ok_olfactory'] = int(row['QC_n_gas_bme688'] > 100 and np.isfinite(row.get('O_BME_dlogR_mean', np.nan)))
    return row



# ---------- final table builder / feature groups ----------

META_COLS = ['dataset','session','obj_id','obj_name','label','group','trial_id','variant','trial_dir']
QC_PREFIX = 'QC_'
DROP_MLREADY_DEFAULT = ['T_contact_cy_mean', 'T_contact_centroid_var', 'A_decay_tau_s']

FEATURE_GROUPS_STRICT = {
    'T': T_KEYS,
    'A': A_KEYS,
    'O': O_KEYS,
    'TA': TA_KEYS,
    'TO': TO_KEYS,
    'AO': AO_KEYS,
    'QC_O_not_for_classifier': QC_O_KEYS,
    'recommended_combo_columns': {
        'T': ['T'],
        'A': ['A'],
        'O': ['O'],
        'T+A': ['T', 'A', 'TA'],
        'T+O': ['T', 'O', 'TO'],
        'A+O': ['A', 'O', 'AO'],
        'T+A+O': ['T', 'A', 'O', 'TA', 'TO', 'AO'],
    },
    'dropped_from_mlready_due_to_missingness': DROP_MLREADY_DEFAULT,
}

# Legacy aliases for scripts that still expect T/A/G/X (e.g. C-model full T+A+G).
# NOTE: old run_ml.py adds X to every >=2 modality combo, so strict ablation should
# use FEATURE_GROUPS_STRICT + combo_feature_columns(), not the legacy FEATURE_GROUPS.
FEATURE_GROUPS = {
    'T': T_KEYS,
    'A': A_KEYS,
    'G': O_KEYS,          # legacy name: G == olfactory O
    'X': X_KEYS,          # all pairwise cross features; OK for full T+A+G, not strict partial ablation
}


def combo_feature_columns(combo: str, *, strict: bool = True) -> List[str]:
    """Return classifier input columns for a modality combo.

    strict=True uses the final no-leakage policy:
      T+A   -> T + A + TA
      T+O/G -> T + O + TO
      A+O/G -> A + O + AO
      Full  -> T + A + O + TA + TO + AO
    """
    combo = combo.replace('G', 'O')
    if not strict:
        mods = combo.split('+')
        cols: List[str] = []
        for m in mods:
            cols += FEATURE_GROUPS_STRICT[m]
        if len(mods) >= 2:
            cols += X_KEYS
        return cols
    group_names = FEATURE_GROUPS_STRICT['recommended_combo_columns'][combo]
    cols: List[str] = []
    for g in group_names:
        cols += FEATURE_GROUPS_STRICT[g]
    return cols


def detect_stage_from_root(root: Path) -> str:
    names = [p.name.lower() for p in root.glob('*') if p.is_dir()]
    joined = ' '.join(names)
    if any(x in joined for x in ['glass_empty', 'glass_ethanol', 'glass_acetone',
                                  'ceramic_empty', 'ceramic_ethanol', 'ceramic_acetone',
                                  'plastic_empty', 'plastic_ethanol', 'plastic_acetone']):
        return 'stage1'
    if any(x in joined for x in ['orange', 'apple', 'pear', 'grapefruit', 'pingpong', 'golf', 'tennis', 'baseball']):
        return 'stage2a'
    if any(x in joined for x in ['skin', 'dragonskin', 'chicken']):
        return 'stage2b'
    return 'stage2a'


def normalize_stage(stage: str, root: Path) -> str:
    if stage in (None, 'auto'):
        return detect_stage_from_root(root)
    raw = str(stage).lower()
    compact = raw.replace('stage', '')
    if compact in ('1', 'run1') or raw in ('stage1', 'stage1_run1'):
        return 'stage1'
    if compact in ('2a', 'a'):
        return 'stage2a'
    if compact in ('2b', 'b'):
        return 'stage2b'
    if raw in ('stage2a', 'stage2b'):
        return raw
    raise ValueError(f'unknown stage: {stage}')


def find_trials(root: Path, pattern: str) -> List[Path]:
    trials = sorted(p for p in root.glob(pattern) if p.is_dir())
    # common fallback: root/obj*/trial_* or root/*/*/trial_*
    if not trials:
        trials = sorted(p for p in root.glob('**/trial_*') if p.is_dir())
    return trials


def reorder_and_prepare(df: pd.DataFrame, *, ml_ready: bool = False) -> pd.DataFrame:
    qc_cols = [c for c in df.columns if c.startswith(QC_PREFIX)]
    feature_cols = T_KEYS + A_KEYS + O_KEYS + TA_KEYS + TO_KEYS + AO_KEYS
    ordered = [c for c in META_COLS if c in df.columns] + qc_cols + feature_cols
    for c in ordered:
        if c not in df.columns:
            df[c] = np.nan
    out = df[ordered + [c for c in df.columns if c not in ordered]].copy()
    if ml_ready:
        out = out.drop(columns=[c for c in DROP_MLREADY_DEFAULT if c in out.columns], errors='ignore')
    return out


def build(root: Path, stage: str = 'auto', out_path: Path | None = None, *,
          pattern: str = '**/trial_*', ml_ready: bool = False) -> pd.DataFrame:
    """Build a final redesigned feature table from a raw data root."""
    stage_norm = normalize_stage(stage, root)
    trials = find_trials(root, pattern)
    if not trials:
        raise FileNotFoundError(f'No trial folders found under {root} with pattern {pattern}')
    rows = []
    for i, tr in enumerate(trials, 1):
        try:
            rows.append(extract_trial(tr, stage_norm))
        except Exception as e:  # keep extraction robust and report problem row
            rows.append({'dataset': stage_norm, 'trial_dir': str(tr), 'ERROR': str(e)})
        if i % 50 == 0:
            print(f'{stage_norm}: {i}/{len(trials)}')
    df = reorder_and_prepare(pd.DataFrame(rows), ml_ready=ml_ready)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def write_groups_json(path: Path, *, ml_ready: bool = False) -> None:
    groups = json.loads(json.dumps(FEATURE_GROUPS_STRICT, ensure_ascii=False))
    if ml_ready:
        drop = set(DROP_MLREADY_DEFAULT)
        for g in ['T', 'A', 'O', 'TA', 'TO', 'AO']:
            groups[g] = [c for c in groups[g] if c not in drop]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding='utf-8')


def print_summary(df: pd.DataFrame, out_path: Path, *, ml_ready: bool) -> None:
    feature_cols = [c for c in T_KEYS + A_KEYS + O_KEYS + TA_KEYS + TO_KEYS + AO_KEYS if c in df.columns]
    print(f'\n저장: {out_path}  ({len(df)} trials × {len(feature_cols)} features + meta/QC)')
    if 'ERROR' in df.columns and df['ERROR'].notna().any():
        print('\n⚠ 추출 오류 row:')
        print(df.loc[df['ERROR'].notna(), ['trial_dir', 'ERROR']].to_string(index=False))
    # QC summary
    qc_cols = [c for c in ['QC_ok_tactile', 'QC_ok_acoustic', 'QC_ok_olfactory'] if c in df.columns]
    if qc_cols:
        print('\nQC pass counts:')
        for c in qc_cols:
            print(f'  {c}: {int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())}/{len(df)}')
    na = df[feature_cols].isna().sum().sort_values(ascending=False)
    bad = na[na > 0]
    if len(bad):
        print('\n결측 있는 feature 상위 20개:')
        for k, v in bad.head(20).items():
            print(f'  {k}: {int(v)}/{len(df)} NaN')
    else:
        print('\n✓ feature 결측 없음')
    if ml_ready:
        print(f'\nML-ready mode: dropped {DROP_MLREADY_DEFAULT}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Final redesigned strict-modality feature extractor for Stage1/Stage2 raw trials')
    ap.add_argument('--data-root', required=True, help='Raw trial root containing obj*/trial_* folders')
    ap.add_argument('--out', default='feature_table_stage2_final.csv', help='Output CSV path')
    ap.add_argument('--stage', choices=['auto', '1', 'stage1', 'stage1_run1', '2a', '2b', 'stage2a', 'stage2b'], default='auto')
    ap.add_argument('--glob', default='**/trial_*', help='Trial folder glob pattern under data-root')
    ap.add_argument('--ml-ready', action='store_true', help='Drop high-missingness diagnostic columns')
    ap.add_argument('--groups-out', default=None, help='Optional JSON path for strict feature groups')
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    out_path = Path(args.out)
    df = build(root, args.stage, out_path, pattern=args.glob, ml_ready=args.ml_ready)
    if args.groups_out:
        write_groups_json(Path(args.groups_out), ml_ready=args.ml_ready)
        print(f'groups 저장: {args.groups_out}')
    print_summary(df, out_path, ml_ready=args.ml_ready)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
