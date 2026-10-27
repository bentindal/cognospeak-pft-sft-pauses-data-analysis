from __future__ import annotations

import re
from typing import Dict

import numpy as np
import pandas as pd


def base_recording_from_key(key: str) -> str:
    base = key.split("__")[0].split("_")[0]
    match = re.match(r"^(Process-(?:rec|test)-)(\d+)$", base)
    if not match:
        return base
    prefix, num = match.group(1), int(match.group(2))
    return f"{prefix}{num:03d}"


def bin_features(df: pd.DataFrame, bin_sec: float = 3.0, max_time: float = 60.0) -> np.ndarray:
    st = pd.to_numeric(df["start_time"], errors="coerce").to_numpy()
    en = pd.to_numeric(df["end_time"], errors="coerce").to_numpy()
    tok = df["token"].astype(str).fillna("").to_numpy()

    ok = np.isfinite(st) & np.isfinite(en)
    st, en, tok = st[ok], en[ok], tok[ok]

    bins = int(max_time // bin_sec)
    out = np.zeros((bins, 6), dtype=np.float32)
    if len(st) == 0:
        return out

    st = np.clip(st, 0, max_time - 1e-6)
    en = np.clip(en, 0, max_time - 1e-6)
    bin_idx = np.floor(st / bin_sec).astype(int)
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    for b in range(bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) == 0:
            continue
        words = [t.strip().lower() for t in tok[idx] if t.strip()]
        out[b, 0] = len(words)
        out[b, 1] = len(set(words))
        out[b, 2] = float(np.maximum(0, en[idx] - st[idx]).sum())

    if len(st) >= 2:
        order = np.argsort(st)
        st2, en2 = st[order], en[order]
        pauses = np.maximum(0.0, st2[1:] - en2[:-1])
        pause_times = st2[1:]
        bin_idx_pause = np.floor(pause_times / bin_sec).astype(int)
        bin_idx_pause = np.clip(bin_idx_pause, 0, bins - 1)

        for b in range(bins):
            p = pauses[bin_idx_pause == b]
            if len(p) == 0:
                continue
            out[b, 3] = float(p.sum())
            out[b, 4] = float(p.mean())
            out[b, 5] = float(p.max())

    return out


def transcript_summary_features(df: pd.DataFrame) -> np.ndarray:
    st = pd.to_numeric(df["start_time"], errors="coerce").to_numpy()
    en = pd.to_numeric(df["end_time"], errors="coerce").to_numpy()
    tok = df["token"].astype(str).fillna("").to_numpy()

    ok = np.isfinite(st) & np.isfinite(en)
    st, en, tok = st[ok], en[ok], tok[ok]

    token_count = float(len(tok))
    if len(st) < 2:
        return np.array([token_count, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    order = np.argsort(st)
    st, en = st[order], en[order]
    pauses = np.maximum(0.0, st[1:] - en[:-1])

    pause_count = float(len(pauses))
    pause_total = float(pauses.sum())
    pause_mean = float(pauses.mean()) if len(pauses) else 0.0
    pause_median = float(np.median(pauses)) if len(pauses) else 0.0
    pause_min = float(pauses.min()) if len(pauses) else 0.0
    pause_max = float(pauses.max()) if len(pauses) else 0.0
    pause_std = float(pauses.std(ddof=1)) if len(pauses) > 1 else 0.0

    if len(pauses) > 1:
        q1, q2, q3 = np.quantile(pauses, [0.25, 0.5, 0.75])
    elif len(pauses) == 1:
        q1 = q2 = q3 = float(pauses[0])
    else:
        q1 = q2 = q3 = 0.0

    return np.array([
        token_count, pause_count, pause_mean, pause_median, pause_min,
        pause_max, pause_std, pause_total, float(q1), float(q2), float(q3)
    ], dtype=np.float32)


def dict_to_feature_tensors(dfs: Dict[str, pd.DataFrame], bin_sec: float = 3.0, max_time: float = 60.0):
    keys = list(dfs.keys())
    x_seq = np.stack([bin_features(dfs[k], bin_sec, max_time) for k in keys], axis=0)
    x_sum = np.stack([transcript_summary_features(dfs[k]) for k in keys], axis=0)
    return keys, x_seq, x_sum
