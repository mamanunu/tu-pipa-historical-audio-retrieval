#!/usr/bin/env python3
"""Build the restricted Tu Pipa field-recording corpus analysis.

The source recordings are read only and treated as mono recordings throughout
the analysis. The decoder produces one analysis channel, and all derived clips
are written as mono WAV files. Derived metadata, acoustic features, PCA scores
and grouped retrieval results are written to ``analysis``. The feature
implementation is deliberately dependency-light and records all parameters so
the analysis can be reproduced and audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCE_ROOT = Path("data/audio")
OUT_ROOT = Path("analysis")
CLIP_ROOT = OUT_ROOT / "segments_audio"
MANUAL_REVIEW_PATH: Path | None = None
MANUAL_CASE_PATH: Path | None = None
WRITE_DERIVED_AUDIO = False
ACCEPT_UNREVIEWED = False
COLLECTOR = "not provided"
PERIOD = "not provided"
LOCATION = "not provided"
SCENE = "field recording"
SOURCE_MEDIUM = "not provided"
PERMISSION_STATUS = "restricted research material"
TARGET_SR = 22050
WINDOW_SECONDS = 4.0
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 50.0
FMAX = 8000.0
RNG_SEED = 20261004
GENRE_HUA = "Hua Diao"
GENRE_BEI = "Bei Gong Diao"
GENRES = (GENRE_BEI, GENRE_HUA)
GENRE_COUNTS = {GENRE_HUA: 45, GENRE_BEI: 9}
SOURCE_FOLDER_ALIASES = {
    GENRE_HUA: ("\u82b1\u8c03", GENRE_HUA),
    GENRE_BEI: ("\u80cc\u5de5\u8c03", GENRE_BEI),
}
UNKNOWN_SOURCE_VALUE = "\u672a\u77e5"
PENDING_SOURCE_VALUE = "\u5f85\u6838"


def run(cmd: Sequence[str], *, text: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=text)


def ffprobe_json(path: Path) -> Dict:
    p = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(p.stdout)


def decode_audio(path: Path, start: float | None = None, duration: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    # The research corpus contains mono source recordings. ``-ac 1`` preserves
    # a single analysis channel without applying a stereo averaging formula.
    cmd += ["-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "-"]
    p = run(cmd)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)


def decode_wav_full(path: Path) -> np.ndarray:
    return decode_audio(path)


def dbfs(value: float, floor: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(float(value), floor)))


def frame_signal(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0, n_fft), dtype=np.float64)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n_frames = 1 + max(0, (x.size - n_fft) // hop)
    starts = np.arange(n_frames) * hop
    frames = np.stack([x[s:s + n_fft] for s in starts])
    return frames * np.hanning(n_fft)[None, :]


def frame_rms_db(x: np.ndarray) -> np.ndarray:
    frames = frame_signal(x)
    if len(frames) == 0:
        return np.array([-120.0])
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames * frames, axis=1)), 1e-12))


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = np.array([mel_to_hz(m) for m in mels])
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(max(0, left), min(center, fb.shape[1])):
            fb[i, k] = (k - left) / max(center - left, 1)
        for k in range(max(0, center), min(right, fb.shape[1])):
            fb[i, k] = (right - k) / max(right - center, 1)
    return fb


MEL_FB = mel_filterbank(TARGET_SR, N_FFT, N_MELS, FMIN, FMAX)
MEL_DCT = np.cos(np.pi / N_MELS * np.arange(14)[:, None] * (np.arange(N_MELS)[None, :] + 0.5))


def spectral_frame_features(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    frames = frame_signal(x)
    if len(frames) == 0:
        frames = np.zeros((1, N_FFT))
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / TARGET_SR)
    total = np.maximum(np.sum(spec, axis=1), 1e-20)
    centroid = np.sum(spec * freqs[None, :], axis=1) / total
    spread = np.sqrt(np.maximum(np.sum(spec * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / total, 0.0))
    centered = freqs[None, :] - centroid[:, None]
    skew = np.sum(spec * centered ** 3, axis=1) / np.maximum(total * spread ** 3, 1e-20)
    kurt = np.sum(spec * centered ** 4, axis=1) / np.maximum(total * spread ** 4, 1e-20)
    cumsum = np.cumsum(spec, axis=1)
    threshold = 0.85 * total
    roll_idx = np.argmax(cumsum >= threshold[:, None], axis=1)
    rolloff = freqs[roll_idx]
    flatness = np.exp(np.mean(np.log(np.maximum(spec, 1e-20)), axis=1)) / np.maximum(np.mean(spec, axis=1), 1e-20)
    flatness_db = 10.0 * np.log10(np.maximum(flatness, 1e-12))
    rms_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames * frames, axis=1)), 1e-12))
    mel_energy = np.maximum(spec @ MEL_FB.T, 1e-12)
    mfcc = np.log(mel_energy) @ MEL_DCT.T
    # Use coefficients 1..13, matching a common MFCC baseline without the
    # broadband energy coefficient, while the separate RMS feature is retained.
    mfcc = mfcc[:, 1:14]
    values = {
        "centroid": centroid,
        "spread": spread,
        "skewness": skew,
        "kurtosis": kurt,
        "rolloff85": rolloff,
        "flatness_db": flatness_db,
        "loudness_dbfs": rms_db,
    }
    return mfcc, values


def mean_std(v: np.ndarray) -> List[float]:
    return [float(np.mean(v)), float(np.std(v, ddof=0))]


def features_for_clip(x: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    mfcc, spec = spectral_frame_features(x)
    out_a: Dict[str, float] = {}
    out_b: Dict[str, float] = {}
    out_c: Dict[str, float] = {}
    for j in range(mfcc.shape[1]):
        a = mean_std(mfcc[:, j])
        out_a[f"mfcc_{j + 1:02d}_mean"], out_a[f"mfcc_{j + 1:02d}_std"] = a
        out_b[f"mfcc_{j + 1:02d}_mean"], out_b[f"mfcc_{j + 1:02d}_std"] = a
        out_c[f"mfcc_{j + 1:02d}_mean"], out_c[f"mfcc_{j + 1:02d}_std"] = a
    for key in ["centroid", "spread", "skewness", "kurtosis", "rolloff85", "flatness_db"]:
        a = mean_std(spec[key])
        out_b[f"{key}_mean"], out_b[f"{key}_std"] = a
        out_c[f"{key}_mean"], out_c[f"{key}_std"] = a
    a = mean_std(spec["loudness_dbfs"])
    out_c["loudness_dbfs_mean"], out_c["loudness_dbfs_std"] = a
    return out_a, out_b, out_c


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_filename(path: Path, form: str) -> Tuple[str, str]:
    stem = path.stem.rstrip(".")
    parts = list(re.finditer(r"[（(](.*?)[）)]", stem))
    performer = UNKNOWN_SOURCE_VALUE
    title = stem
    if len(parts) >= 2:
        # The final parenthesized group is treated as performer metadata, while
        # earlier parenthesized groups remain part of the source title.
        performer = parts[-1].group(1).strip() or UNKNOWN_SOURCE_VALUE
        title = stem[:parts[-1].start()] + stem[parts[-1].end():]
    elif len(parts) == 1:
        candidate = parts[0].group(1).strip()
        segment_word = "\u7247\u6bb5"
        source_numerals = r"[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u3001]+"
        if segment_word not in candidate and not re.fullmatch(source_numerals, candidate):
            performer = candidate or UNKNOWN_SOURCE_VALUE
            title = stem[:parts[0].start()] + stem[parts[0].end():]
    title = title.strip().rstrip(".")
    title = re.sub(rf"^{re.escape(form)}\s*\d+\s*[.、]\s*", "", title)
    title = re.sub(r"^\d+\s*[.、]\s*", "", title)
    return title or stem, performer or UNKNOWN_SOURCE_VALUE


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def robust_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    iqr = q75 - q25
    safe = np.where(iqr < 1e-12, 1.0, iqr)
    return (x - med) / safe, med, safe


def pca(x: np.ndarray, n_components: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    centered = x - np.mean(x, axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]
    eigen = (s ** 2) / max(1, x.shape[0] - 1)
    explained = eigen / max(float(np.sum(eigen)), 1e-12)
    return scores, explained[:n_components]


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum((b - a[None, :]) ** 2, axis=1), 0.0))


def ranked_indices(distances: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    """Sort by distance, using the stable object ID as a deterministic tie break."""
    return np.lexsort((np.asarray(ids, dtype=str), distances))


def ap_at_k(relevant: List[bool], k: int, total_relevant: int) -> float:
    rel = relevant[:k]
    denominator = min(total_relevant, k)
    if denominator == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, ok in enumerate(rel, 1):
        if ok:
            hits += 1
            score += hits / i
    return score / denominator


def mrr(relevant: List[bool]) -> float:
    for i, ok in enumerate(relevant, 1):
        if ok:
            return 1.0 / i
    return 0.0


def exact_random_metrics(n_candidates: int, n_relevant: int, k: int = 5) -> Dict[str, float]:
    """Exact random-ranking expectations for the truncated top-k metrics."""
    states = [(0, 0, 1.0, [])]  # relevant used, nonrelevant used, probability, labels
    for _ in range(k):
        next_states = []
        for used_rel, used_non, prob, labels in states:
            remaining = n_candidates - used_rel - used_non
            if n_relevant - used_rel > 0:
                p_rel = (n_relevant - used_rel) / remaining
                next_states.append((used_rel + 1, used_non, prob * p_rel, labels + [True]))
            if (n_candidates - n_relevant) - used_non > 0:
                p_non = ((n_candidates - n_relevant) - used_non) / remaining
                next_states.append((used_rel, used_non + 1, prob * p_non, labels + [False]))
        states = next_states
    out = {"p_at_5": 0.0, "hit_rate_at_1": 0.0, "hit_rate_at_5": 0.0, "mrr_at_5": 0.0, "map_at_5": 0.0}
    for _, _, prob, labels in states:
        out["p_at_5"] += prob * float(np.mean(labels))
        out["hit_rate_at_1"] += prob * float(labels[0])
        out["hit_rate_at_5"] += prob * float(any(labels))
        out["mrr_at_5"] += prob * mrr(labels)
        out["map_at_5"] += prob * ap_at_k(labels, k, n_relevant)
    return out


def performer_tokens(value: str) -> set[str]:
    if not value or value == UNKNOWN_SOURCE_VALUE:
        return set()
    list_separator = "\u3001"
    cleaned = (
        value.replace("\uff0c", list_separator)
        .replace(",", list_separator)
        .replace("\u548c", list_separator)
    )
    return {x.strip() for x in cleaned.split(list_separator) if x.strip()}


def bootstrap_ci(values: Sequence[float], group_ids: Sequence[str], n_boot: int = 1000) -> Tuple[float, float]:
    groups = sorted(set(group_ids))
    by_group = {g: [v for v, gg in zip(values, group_ids) if gg == g] for g in groups}
    rng = np.random.default_rng(RNG_SEED + len(groups))
    means = []
    for _ in range(n_boot):
        sample = rng.choice(groups, size=len(groups), replace=True)
        flat = [v for g in sample for v in by_group[g]]
        means.append(float(np.mean(flat)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_macro_ci(
    values: Sequence[float], group_ids: Sequence[str], forms: Sequence[str], n_boot: int = 1000
) -> Tuple[float, float]:
    """Stratified query-recording bootstrap for a two-form macro mean.

    Rankings and the candidate corpus remain fixed. Query recordings are
    resampled with replacement within each traditional category, retaining all
    five query segments for every sampled recording.
    """
    rows = list(zip(values, group_ids, forms))
    rng = np.random.default_rng(RNG_SEED + 207)
    estimates = []
    for _ in range(n_boot):
        form_means = []
        for form in GENRES:
            groups = sorted({g for _, g, f in rows if f == form})
            by_group = {g: [v for v, gg, f in rows if gg == g and f == form] for g in groups}
            sampled = rng.choice(groups, size=len(groups), replace=True)
            form_means.append(float(np.mean([v for g in sampled for v in by_group[g]])))
        estimates.append(float(np.mean(form_means)))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def load_fonts(title_size: int = 24, body_size: int = 18) -> Tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    for font_path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, title_size), ImageFont.truetype(font_path, body_size)
        except Exception:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def plot_pca(
    scores: np.ndarray,
    forms: Sequence[str],
    labels: Sequence[str],
    title: str,
    out: Path,
    explained: Sequence[float] | None = None,
) -> None:
    width, height = 1400, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font, small = load_fonts()
    x = scores[:, 0]
    y = scores[:, 1]
    pad = 90
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    colors = {GENRE_HUA: (45, 103, 180), GENRE_BEI: (210, 87, 55)}
    draw.text((pad, 22), title, fill=(20, 20, 20), font=font)
    draw.line((pad, height - pad, width - 45, height - pad), fill=(100, 100, 100), width=2)
    draw.line((pad, height - pad, pad, 70), fill=(100, 100, 100), width=2)
    for i, (xx, yy, form) in enumerate(zip(x, y, forms)):
        px = pad + (xx - xmin) / dx * (width - pad - 50)
        py = height - pad - (yy - ymin) / dy * (height - pad - 100)
        c = colors.get(form, (100, 100, 100))
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=c, outline=c)
    x_label = "PC1" if explained is None else f"PC1（{explained[0] * 100:.1f}%）"
    y_label = "PC2" if explained is None else f"PC2（{explained[1] * 100:.1f}%）"
    draw.text((width // 2 - 110, height - 45), x_label, fill=(40, 40, 40), font=small)
    draw.text((10, 80), y_label, fill=(40, 40, 40), font=small)
    draw.ellipse((width - 300, 90, width - 284, 106), fill=colors[GENRE_HUA])
    draw.text((width - 275, 86), GENRE_HUA, fill=(30, 30, 30), font=small)
    draw.ellipse((width - 300, 125, width - 284, 141), fill=colors[GENRE_BEI])
    draw.text((width - 275, 121), GENRE_BEI, fill=(30, 30, 30), font=small)
    img.save(out)


def plot_pca_combined(
    panels: Dict[str, Tuple[np.ndarray, np.ndarray]], forms: Sequence[str], out: Path
) -> None:
    """Draw a publication-oriented A/B/C PCA panel for 4-second clips."""
    width, height = 2400, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, body_font = load_fonts(28, 18)
    _, small_font = load_fonts(22, 15)
    colors = {GENRE_HUA: (45, 103, 180), GENRE_BEI: (210, 87, 55)}
    panel_names = {
        "A": "A  MFCC",
        "B": "B  MFCC + spectral shape",
        "C": "C  MFCC + spectral shape + log-RMS",
    }
    draw.text((55, 24), "PCA spaces for the three acoustic representations", fill=(24, 35, 55), font=title_font)
    panel_width = 730
    panel_gap = 45
    top = 92
    bottom = 710
    for panel_index, key in enumerate(["A", "B", "C"]):
        scores, explained = panels[key]
        left = 60 + panel_index * (panel_width + panel_gap)
        right = left + panel_width
        plot_left, plot_right = left + 72, right - 22
        plot_top, plot_bottom = top + 72, bottom - 40
        draw.text((left + 6, top), panel_names[key], fill=(32, 42, 62), font=body_font)
        subtitle = f"PC1 {explained[0] * 100:.1f}%   ·   PC2 {explained[1] * 100:.1f}%"
        draw.text((left + 6, top + 31), subtitle, fill=(80, 88, 104), font=small_font if 'small_font' in locals() else body_font)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(165, 174, 190), width=2)
        x, y = scores[:, 0], scores[:, 1]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
        for xx, yy, form in zip(x, y, forms):
            px = plot_left + (float(xx) - xmin) / dx * (plot_right - plot_left)
            py = plot_bottom - (float(yy) - ymin) / dy * (plot_bottom - plot_top)
            c = colors.get(form, (110, 110, 110))
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=c, outline=(255, 255, 255), width=1)
        x_label = "PC1"
        y_label = "PC2"
        draw.text((left + 330, bottom - 24), x_label, fill=(50, 58, 72), font=body_font)
        draw.text((left + 18, plot_top + 8), y_label, fill=(50, 58, 72), font=body_font)
    legend_y = 752
    draw.ellipse((560, legend_y, 574, legend_y + 14), fill=colors[GENRE_HUA])
    draw.text((585, legend_y - 7), GENRE_HUA, fill=(35, 40, 50), font=body_font)
    draw.ellipse((720, legend_y, 734, legend_y + 14), fill=colors[GENRE_BEI])
    draw.text((745, legend_y - 7), GENRE_BEI, fill=(35, 40, 50), font=body_font)
    draw.text(
        (930, legend_y - 7),
        "Each point is one four-second segment; five segments from a recording share its work-level genre label.",
        fill=(75, 82, 95),
        font=body_font,
    )
    img.save(out)


def plot_corpus_workflow(out: Path) -> None:
    """Show corpus entities, analysis flow, traceability, and access layers."""
    width, height = 2400, 1280
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, body_font = load_fonts(30, 21)
    _, small_font = load_fonts(22, 17)
    draw.text((70, 40), "Auditable workflow for restricted historical field recordings", fill=(24, 35, 55), font=title_font)
    boxes = [
        ("Source verification", "Catalog and publication\nFile checksums\nAccess permissions"),
        ("Quality audit", "Decoding and levels\nNoise and silence\nListening review"),
        ("Corpus linkage", "Works and genres\nPerformers and collector\nPeriod and region"),
        ("Segment review", "Five segments/recording\nStart and end times\nOverlap and listening"),
        ("Acoustic features", "A: MFCC\nB: + spectral shape\nC: + log-RMS"),
        ("Retrieval robustness", "Prototype and controls\nQuality sensitivity\nTop-5 stability"),
        ("Musicological review", "Works and performers\nListening and uncertainty\nTiered outputs"),
    ]
    box_w, box_h, gap = 285, 260, 43
    left0, top = 70, 185
    for i, (heading, detail) in enumerate(boxes):
        left = left0 + i * (box_w + gap)
        right = left + box_w
        draw.rounded_rectangle((left, top, right, top + box_h), radius=18, fill=(242, 246, 251), outline=(78, 103, 139), width=3)
        draw.text((left + 22, top + 24), heading, fill=(30, 55, 90), font=body_font)
        draw.multiline_text((left + 22, top + 82), detail, fill=(55, 66, 82), font=small_font, spacing=12)
        if i < len(boxes) - 1:
            x1, x2, y = right + 8, right + gap - 8, top + box_h // 2
            draw.line((x1, y, x2, y), fill=(91, 105, 125), width=4)
            draw.polygon([(x2, y), (x2 - 15, y - 9), (x2 - 15, y + 9)], fill=(91, 105, 125))
    draw.text((70, 510), "Traceability: recording ID -> segment ID and timestamp -> features/retrieval -> work, publication, and source metadata", fill=(56, 66, 82), font=body_font)

    draw.text((70, 610), "Access tiers", fill=(24, 35, 55), font=title_font)
    layers = [
        ("Tier 1  Restricted source audio", "Tape transfers and audible clips; available only to authorized researchers and not distributed with the paper.", (255, 238, 235), (170, 70, 55)),
        ("Tier 2  Internal research data", "Segment times, performer links, quality flags, review records, and traceable feature matrices.", (255, 247, 224), (165, 112, 32)),
        ("Tier 3  Shareable derived outputs", "Field definitions, parameters, code, aggregate statistics, and authorized non-reconstructive results.", (233, 246, 238), (54, 120, 78)),
    ]
    layer_top = 700
    for i, (heading, detail, fill, outline) in enumerate(layers):
        y1 = layer_top + i * 150
        draw.rounded_rectangle((110, y1, width - 110, y1 + 112), radius=16, fill=fill, outline=outline, width=3)
        draw.text((145, y1 + 22), heading, fill=outline, font=body_font)
        draw.text((690, y1 + 24), detail, fill=(55, 62, 72), font=small_font)
    draw.text((70, 1195), "Computability depends on linked data and traceability; robustness checks and musicological review define the limits of interpretation.", fill=(75, 82, 95), font=small_font)
    img.save(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Tu Pipa corpus analysis on an authorized local copy of the recordings."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Directory containing two immediate subdirectories of mono WAV files: "
            "one with 45 recordings and one with 9."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis"),
        help="Directory for generated private analysis outputs (default: ./analysis).",
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        help="Private JSON file containing the manually reviewed segment decisions.",
    )
    parser.add_argument(
        "--manual-cases",
        type=Path,
        help="Optional private JSON file containing case-listening notes.",
    )
    parser.add_argument(
        "--write-derived-audio",
        action="store_true",
        help="Write 4-second WAV clips. Disabled by default because the source material is restricted.",
    )
    parser.add_argument(
        "--accept-unreviewed",
        action="store_true",
        help="Accept fixed-position samples without a review file. Do not use this flag to reproduce the paper.",
    )
    parser.add_argument("--collector", default=COLLECTOR, help="Collector metadata written to private outputs.")
    parser.add_argument("--period", default=PERIOD, help="Recording period written to private outputs.")
    parser.add_argument("--location", default=LOCATION, help="Location metadata written to private outputs.")
    parser.add_argument("--scene", default=SCENE, help="Recording-scene metadata written to private outputs.")
    parser.add_argument("--source-medium", default=SOURCE_MEDIUM, help="Source-medium description.")
    parser.add_argument("--permission-status", default=PERMISSION_STATUS, help="Local access status description.")
    return parser.parse_args()


def main() -> None:
    global SOURCE_ROOT, OUT_ROOT, CLIP_ROOT, MANUAL_REVIEW_PATH, MANUAL_CASE_PATH
    global WRITE_DERIVED_AUDIO, ACCEPT_UNREVIEWED
    global COLLECTOR, PERIOD, LOCATION, SCENE, SOURCE_MEDIUM, PERMISSION_STATUS

    args = parse_args()
    SOURCE_ROOT = args.source_root.expanduser().resolve()
    OUT_ROOT = args.output_root.expanduser().resolve()
    CLIP_ROOT = OUT_ROOT / "segments_audio"
    MANUAL_REVIEW_PATH = args.manual_review.expanduser().resolve() if args.manual_review else None
    MANUAL_CASE_PATH = args.manual_cases.expanduser().resolve() if args.manual_cases else None
    WRITE_DERIVED_AUDIO = bool(args.write_derived_audio)
    ACCEPT_UNREVIEWED = bool(args.accept_unreviewed)
    COLLECTOR = args.collector
    PERIOD = args.period
    LOCATION = args.location
    SCENE = args.scene
    SOURCE_MEDIUM = args.source_medium
    PERMISSION_STATUS = args.permission_status

    if not SOURCE_ROOT.is_dir():
        raise RuntimeError(f"Source directory does not exist: {SOURCE_ROOT}")
    if MANUAL_REVIEW_PATH is None and not ACCEPT_UNREVIEWED:
        raise RuntimeError(
            "A manually reviewed segment file is required to reproduce the paper. "
            "Use --manual-review PATH, or --accept-unreviewed only for a demonstration on your own data."
        )
    if MANUAL_REVIEW_PATH is not None and not MANUAL_REVIEW_PATH.is_file():
        raise RuntimeError(f"Manual review file does not exist: {MANUAL_REVIEW_PATH}")
    if MANUAL_CASE_PATH is not None and not MANUAL_CASE_PATH.is_file():
        raise RuntimeError(f"Manual case file does not exist: {MANUAL_CASE_PATH}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if WRITE_DERIVED_AUDIO:
        CLIP_ROOT.mkdir(parents=True, exist_ok=True)
    manual_review = {}
    if MANUAL_REVIEW_PATH and MANUAL_REVIEW_PATH.exists():
        try:
            manual_rows = json.loads(MANUAL_REVIEW_PATH.read_text(encoding="utf-8"))
            manual_review = {
                row.get("segment_id"): row
                for row in manual_rows
                if row.get("segment_id")
            }
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Could not read manual review file: {MANUAL_REVIEW_PATH}") from exc
    manual_case_interpretations = {}
    if MANUAL_CASE_PATH and MANUAL_CASE_PATH.exists():
        try:
            case_notes = json.loads(MANUAL_CASE_PATH.read_text(encoding="utf-8"))
            manual_case_interpretations = {
                (row.get("case_type"), row.get("query_segment_id"), row.get("result_recording_id")): row.get("manual_interpretation", "")
                for row in case_notes
            }
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Could not read manual case file: {MANUAL_CASE_PATH}") from exc
    genre_groups = {}
    for genre in GENRES:
        matches = [
            SOURCE_ROOT / alias
            for alias in SOURCE_FOLDER_ALIASES[genre]
            if (SOURCE_ROOT / alias).is_dir()
        ]
        if len(matches) != 1:
            aliases = ", ".join(repr(alias) for alias in SOURCE_FOLDER_ALIASES[genre])
            raise RuntimeError(
                f"Expected exactly one source folder for {genre} using one of these aliases: {aliases}. "
                f"Found {len(matches)}."
            )
        folder = matches[0]
        paths = sorted(folder.glob("*.wav"))
        expected_count = GENRE_COUNTS[genre]
        if len(paths) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {genre} WAV files in {folder.name!r}, found {len(paths)}."
            )
        genre_groups[genre] = (folder, paths)
    file_rows = [
        (path, genre, folder.name)
        for genre in GENRES
        for folder, paths in [genre_groups[genre]]
        for path in paths
    ]
    files = [path for path, _, _ in file_rows]
    form_counts = {genre: len(genre_groups[genre][1]) for genre in GENRES}
    if form_counts != GENRE_COUNTS:
        raise RuntimeError(f"Unexpected genre counts: {form_counts}")

    recordings: List[Dict] = []
    audio_cache: Dict[str, np.ndarray] = {}
    segments: List[Dict] = []
    feature_rows = {"A": [], "B": [], "C": []}
    quality_rows: Dict[str, Dict] = {}

    for index, (path, form, source_form_name) in enumerate(file_rows, 1):
        recording_id = f"TP{index:03d}"
        title, performer = parse_filename(path, source_form_name)
        probe = ffprobe_json(path)
        fmt = probe.get("format", {})
        stream = (probe.get("streams") or [{}])[0]
        source_channels = int(stream.get("channels") or 0)
        if source_channels != 1:
            raise RuntimeError(
                f"Expected a mono source recording, found {source_channels} channels: {path.name}"
            )
        duration = float(fmt.get("duration") or 0.0)
        audio = decode_wav_full(path)
        audio_cache[recording_id] = audio
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        frame_db = frame_rms_db(audio)
        # This is the 10th percentile of frame-level RMS, not a measurement of
        # background noise or signal-to-noise ratio. Quiet music may contribute.
        low_energy_p10 = float(np.percentile(frame_db, 10))
        silence_ratio = float(np.mean(frame_db < -45.0))
        rms_db = dbfs(rms)
        peak_db = dbfs(peak)
        flags = []
        if rms_db < -35.0:
            flags.append("low_level")
        if silence_ratio > 0.25:
            flags.append("long_silence")
        quality_flag = ";".join(flags) if flags else "normal_for_archive_audio"
        quality_rows[recording_id] = {
            "rms_dbfs": rms_db, "peak_dbfs": peak_db,
            "low_energy_frame_p10_dbfs": low_energy_p10, "silence_ratio_below_-45db": silence_ratio,
            "quality_flag": quality_flag,
        }
        recordings.append({
            "recording_id": recording_id,
            "form": form,
            "category_label_source": "Source-corpus category directory, checked work by work against the relevant published anthology entries",
            "category_label_level": "work level",
            "title": title,
            "performer": performer,
            "collector": COLLECTOR,
            "period": PERIOD,
            "location": LOCATION,
            "scene": SCENE,
            "source_medium": SOURCE_MEDIUM,
            "source_path_relative": str(path.relative_to(SOURCE_ROOT)),
            "duration_s": round(duration, 6),
            "sample_rate_hz": stream.get("sample_rate", "44100"),
            "channels": 1,
            "codec": stream.get("codec_name", "unknown"),
            "bits_per_sample": stream.get("bits_per_sample", ""),
            "bits_per_raw_sample": stream.get("bits_per_raw_sample", ""),
            "permission_status": PERMISSION_STATUS,
            "checksum_sha256": sha256(path),
            "rms_dbfs": round(rms_db, 4),
            "peak_dbfs": round(peak_db, 4),
            "low_energy_frame_p10_dbfs": round(low_energy_p10, 4),
            "silence_ratio_below_-45db": round(silence_ratio, 6),
            "quality_flag": quality_flag,
        })

        valid_start = 0.02 * duration
        valid_end = 0.98 * duration
        target_seconds = [0.20, 0.35, 0.50, 0.65, 0.80]
        for part, frac in enumerate(target_seconds, 1):
            segment_id = f"{recording_id}_S{part:02d}"
            center = valid_start + frac * (valid_end - valid_start)
            default_start = max(valid_start, min(center - WINDOW_SECONDS / 2.0, valid_end - WINDOW_SECONDS))
            review_row = manual_review.get(segment_id, {})
            if ACCEPT_UNREVIEWED and not review_row:
                review_row = {
                    "manual_review_status": "keep",
                    "manual_review_note": "Accepted without manual review for demonstration only.",
                }
            replacement_start = review_row.get("replacement_start_s")
            replacement_end = review_row.get("replacement_end_s")
            if replacement_start is not None:
                start = float(replacement_start)
                if replacement_end is not None and not math.isclose(
                    float(replacement_end) - start, WINDOW_SECONDS, abs_tol=1e-6
                ):
                    raise RuntimeError(f"Replacement interval for {segment_id} must be {WINDOW_SECONDS} seconds")
                if start < 0 or start + WINDOW_SECONDS > duration:
                    raise RuntimeError(f"Replacement interval for {segment_id} is outside the source recording")
                sampling_method = "manual_replacement"
            else:
                start = default_start
                sampling_method = "fixed_fraction"
            clip = audio[int(round(start * TARGET_SR)):int(round((start + WINDOW_SECONDS) * TARGET_SR))]
            if clip.size < int(WINDOW_SECONDS * TARGET_SR):
                clip = np.pad(clip, (0, int(WINDOW_SECONDS * TARGET_SR) - clip.size))
            # Mark windows that are unusually quiet. Do not silently replace them:
            # this preserves the sampling rule and makes review explicit.
            clip_rms_db = dbfs(float(np.sqrt(np.mean(clip * clip))))
            clip_path = CLIP_ROOT / f"{segment_id}.wav"
            if WRITE_DERIVED_AUDIO:
                pcm = np.clip(clip, -1.0, 1.0)
                pcm_i16 = (pcm * 32767.0).astype("<i2")
                with wave.open(str(clip_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(TARGET_SR)
                    wf.writeframes(pcm_i16.tobytes())
            qflag = "review_low_energy" if clip_rms_db < -40 else "candidate"
            segment = {
                "segment_id": segment_id,
                "recording_id": recording_id,
                "form": form,
                "category_label_source": "Inherited from the verified traditional genre of the work associated with the source recording",
                "category_label_level": "segment-level inheritance of a work-level label",
                "title": title,
                "performer": performer,
                "source_file": str(path.relative_to(SOURCE_ROOT)),
                "start_s": round(float(start), 6),
                "end_s": round(float(start + WINDOW_SECONDS), 6),
                "sample_rate_hz": TARGET_SR,
                "channels": 1,
                "window_s": WINDOW_SECONDS,
                "sampling_fraction": frac,
                "sampling_method": sampling_method,
                "segment_rms_dbfs": round(clip_rms_db, 4),
                "segment_quality_flag": qflag,
                "manual_review_status": review_row.get("manual_review_status", "needs_review"),
                "manual_review_note": review_row.get("manual_review_note", ""),
                "derived_clip_path": str(clip_path.relative_to(OUT_ROOT)) if WRITE_DERIVED_AUDIO else "",
            }
            segments.append(segment)
            a, b, c = features_for_clip(clip)
            common = {
                "segment_id": segment_id, "recording_id": recording_id,
                "form": form, "title": title, "performer": performer,
                "segment_rms_dbfs": round(clip_rms_db, 4),
            }
            feature_rows["A"].append({**common, **a})
            feature_rows["B"].append({**common, **b})
            feature_rows["C"].append({**common, **c})

    # Use only manually accepted clips for the final computational corpus.
    # A replace status without a new timestamp remains excluded until the
    # reviewer supplies a replacement interval. The full candidate list and
    # statuses remain available in segments.csv for auditability.
    unreviewed_ids = sorted(
        row["segment_id"] for row in segments
        if row.get("manual_review_status") == "needs_review"
    )
    if unreviewed_ids:
        raise RuntimeError(
            f"{len(unreviewed_ids)} segments have not been manually reviewed. "
            "Provide a complete review file before running the paper analysis."
        )

    accepted_ids = {
        row["segment_id"]
        for row in segments
        if row.get("manual_review_status") == "keep"
    }
    invalid_statuses = sorted({
        str(row.get("manual_review_status", ""))
        for row in segments
        if row.get("manual_review_status") != "keep"
    })
    if invalid_statuses or len(accepted_ids) != 270:
        raise RuntimeError(
            "Reproducing the published analysis requires exactly 270 manually accepted segments "
            f"(five per recording). Found {len(accepted_ids)} accepted segments and statuses {invalid_statuses}."
        )
    accepted_per_recording = {
        recording_id: sum(
            row["recording_id"] == recording_id and row["segment_id"] in accepted_ids
            for row in segments
        )
        for recording_id in {row["recording_id"] for row in segments}
    }
    if set(accepted_per_recording.values()) != {5}:
        raise RuntimeError(f"Expected five accepted segments per recording: {accepted_per_recording}")
    for key in ["A", "B", "C"]:
        feature_rows[key] = [row for row in feature_rows[key] if row["segment_id"] in accepted_ids]

    rec_fields = list(recordings[0].keys())
    seg_fields = list(segments[0].keys())
    write_csv(OUT_ROOT / "recordings.csv", recordings, rec_fields)
    write_csv(OUT_ROOT / "segments.csv", segments, seg_fields)
    # Make the label-provenance chain explicit. The directory source and the
    # publication-level title correspondence have both been checked.
    label_provenance_rows = []
    for row in recordings:
        label_provenance_rows.append({
            "recording_id": row["recording_id"],
            "title": row["title"],
            "genre_label": row["form"],
            "label_source": "Source-corpus category directory, checked work by work against the relevant published anthology entries",
            "verification_status": "Source catalog and publication correspondence verified",
            "notes": "Each recording corresponds to one identifiable work and one traditional genre. Segments inherit the work-level label; publication correspondence does not imply that every performed detail is identical to the notated source.",
        })
    write_csv(
        OUT_ROOT / "label_provenance.csv",
        label_provenance_rows,
        ["recording_id", "title", "genre_label", "label_source", "verification_status", "notes"],
    )

    completeness_specs = [
        ("Work title", "title", "Every recording has a work title"),
        ("Traditional genre", "form", "Verified against the source category directory and published work correspondence"),
        ("Performer", "performer", "Unknown values are not inferred"),
        ("Collector", "collector", "Provided as a run parameter"),
        ("Recording period", "period", "Provided as a run parameter"),
        ("Recording region", "regional_location", "Provided as a run parameter; more specific locations can be managed separately"),
        ("Specific village", "specific_village", "Pending verification; the program does not infer a location"),
        ("Performance setting", "scene", "Provided as a run parameter"),
        ("Permission status", "permission_status", "Determined by the authorization applying to the local material"),
        ("Quality flag", "quality_flag", "Generated by the consistent quality checks"),
    ]
    completeness_rows = []
    for display_name, key, note in completeness_specs:
        if key == "specific_village":
            known, unknown, pending = 0, 0, len(recordings)
        elif key == "regional_location":
            values = [str(r.get("location", "")).strip() for r in recordings]
            missing = {"", UNKNOWN_SOURCE_VALUE, "not provided"}
            known = sum(value not in missing and PENDING_SOURCE_VALUE not in value for value in values)
            pending = sum(PENDING_SOURCE_VALUE in value for value in values)
            unknown = len(values) - known - pending
        else:
            values = [str(r.get(key, "")).strip() for r in recordings]
            pending = sum(PENDING_SOURCE_VALUE in value for value in values)
            known = sum(
                bool(value)
                and value != "not provided"
                and UNKNOWN_SOURCE_VALUE not in value
                and PENDING_SOURCE_VALUE not in value
                for value in values
            )
            unknown = len(values) - known - pending
        completeness_rows.append({
            "metadata_field": display_name,
            "record_count": len(recordings),
            "known_count": known,
            "unknown_count": unknown,
            "pending_verification_count": pending,
            "note": note,
        })
    write_csv(
        OUT_ROOT / "metadata_completeness.csv",
        completeness_rows,
        ["metadata_field", "record_count", "known_count", "unknown_count", "pending_verification_count", "note"],
    )
    feature_fields = {k: list(feature_rows[k][0].keys()) for k in feature_rows}
    for k in ["A", "B", "C"]:
        write_csv(OUT_ROOT / {"A": "features_A_mfcc.csv", "B": "features_B_spectral.csv", "C": "features_C_loudness.csv"}[k], feature_rows[k], feature_fields[k])

    # Audit actual segment intervals after all manual replacements. Overlap is
    # assessed from the stored timestamps rather than inferred from duration.
    overlap_rows: List[Dict] = []
    by_recording_segments: Dict[str, List[Dict]] = {}
    for row in segments:
        if row.get("manual_review_status") == "keep":
            by_recording_segments.setdefault(row["recording_id"], []).append(row)
    for rid, rows in by_recording_segments.items():
        ordered = sorted(rows, key=lambda row: float(row["start_s"]))
        for left, right in zip(ordered, ordered[1:]):
            overlap_s = max(0.0, float(left["end_s"]) - float(right["start_s"]))
            if overlap_s > 1e-9:
                overlap_rows.append({
                    "recording_id": rid,
                    "left_segment_id": left["segment_id"],
                    "right_segment_id": right["segment_id"],
                    "left_start_s": left["start_s"],
                    "left_end_s": left["end_s"],
                    "right_start_s": right["start_s"],
                    "right_end_s": right["end_s"],
                    "overlap_s": overlap_s,
                })
    overlap_fields = [
        "recording_id", "left_segment_id", "right_segment_id", "left_start_s",
        "left_end_s", "right_start_s", "right_end_s", "overlap_s",
    ]
    write_csv(OUT_ROOT / "segment_overlap_audit.csv", overlap_rows, overlap_fields)

    # Record both the original automatically sampled interval and the final
    # reviewed replacement interval for exact reconstruction of the corpus.
    replacement_audit_rows: List[Dict] = []
    recording_duration = {row["recording_id"]: float(row["duration_s"]) for row in recordings}
    for row in segments:
        if row.get("sampling_method") != "manual_replacement":
            continue
        duration = recording_duration[row["recording_id"]]
        valid_start = 0.02 * duration
        valid_end = 0.98 * duration
        frac = float(row["sampling_fraction"])
        center = valid_start + frac * (valid_end - valid_start)
        original_start = max(valid_start, min(center - WINDOW_SECONDS / 2.0, valid_end - WINDOW_SECONDS))
        replacement_audit_rows.append({
            "segment_id": row["segment_id"],
            "recording_id": row["recording_id"],
            "title": row["title"],
            "original_start_s": original_start,
            "original_end_s": original_start + WINDOW_SECONDS,
            "final_start_s": row["start_s"],
            "final_end_s": row["end_s"],
            "replacement_reason_and_review": row["manual_review_note"],
        })
    replacement_audit_fields = [
        "segment_id", "recording_id", "title", "original_start_s", "original_end_s",
        "final_start_s", "final_end_s", "replacement_reason_and_review",
    ]
    write_csv(OUT_ROOT / "replacement_audit.csv", replacement_audit_rows, replacement_audit_fields)

    manifest = {
        "source_root": "restricted local input (path omitted)", "core_recordings": len(recordings),
        "candidate_segments": len(segments), "accepted_segments": len(accepted_ids),
        "target_sample_rate_hz": TARGET_SR, "window_s": WINDOW_SECONDS,
        "recording_channel_count": 1,
        "channel_policy": "Require mono source recordings and decode one analysis channel; derived clips are optional and disabled by default.",
        "feature_frame_n_fft": N_FFT, "feature_hop_samples": HOP,
        "mel_bands": N_MELS, "mel_fmin_hz": FMIN, "mel_fmax_hz": FMAX,
        "mfcc_coefficients": "C1-C13 (14 DCT rows generated; C0 excluded)",
        "analysis_window": "Hann window implemented with numpy.hanning",
        "spectrum": "squared magnitude rFFT (power spectrum)",
        "spectral_descriptors": "centroid, spread, skewness, kurtosis, rolloff85 and flatness are computed from the power spectrum; flatness is converted to 10*log10",
        "spectral_rolloff": "frequency below which cumulative power reaches 85%",
        "rms": "20*log10(max(sqrt(mean(Hann-windowed-frame^2)), 1e-12))",
        "distance": "Euclidean distance after per-dimension median/IQR scaling; exact ties ordered by stable ID",
        "zero_iqr_rule": "IQR values below 1e-12 are replaced by 1.0",
        "standardization": "median/IQR robust scaling fitted on the current closed corpus/full 270-clip reference set for descriptive PCA and corpus-internal retrieval; not a train/test generalization estimate",
        "pca": "centered NumPy SVD; scores=U*S; variance ratios from singular values; no whitening",
        "sampling_positions": [0.20, 0.35, 0.50, 0.65, 0.80],
        "feature_note": "Dependency-light Python/NumPy implementation for reproducible corpus-internal analysis; all feature parameters are recorded in this manifest.",
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "ffmpeg": run(["ffmpeg", "-version"], text=True).stdout.splitlines()[0],
        },
        "category_label_rule": "Hua Diao and Bei Gong Diao follow each work's established traditional genre. The source catalog and publication correspondence were checked work by work; four-second segments inherit the verified work-level label.",
        "recording_title_count": len({row["title"] for row in recordings}),
        "duplicate_titles": sorted(title for title in {row["title"] for row in recordings} if sum(r["title"] == title for r in recordings) > 1),
        "segment_overlap_pairs": len(overlap_rows),
        "overlap_sensitivity_unit": "Entire recording; remove all queries, prototypes, and candidate positions from recordings with overlapping windows",
        "main_task_name": "Cross-work segment-to-recording-prototype nearest-neighbor retrieval",
        "access_layers": [
            "Restricted source audio: authorized for research analysis but not publicly distributed",
            "Internal research data: time indexes, metadata, quality flags, and feature matrices",
            "Shareable derived outputs: parameters, code, aggregate statistics, and authorized non-reconstructive results",
        ],
        "manual_review_source": "private review file supplied" if MANUAL_REVIEW_PATH else None,
        "excluded_without_replacement": sorted(
            row["segment_id"] for row in segments
            if row.get("manual_review_status") in {"replace", "exclude"}
        ),
        "manual_replacement_segments": sorted(
            row["segment_id"] for row in segments
            if row.get("sampling_method") == "manual_replacement"
        ),
        "created_at": datetime.now(timezone.utc).date().isoformat(), "random_seed": RNG_SEED,
    }
    (OUT_ROOT / "processing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "data_dictionary.txt").write_text(
        "Data dictionary\n\n"
        "recordings.csv: one row per source recording, including provenance, traditional genre, label basis, work, performer, file format, permissions, and quality fields; channels records the mono analysis channel.\n"
        "segments.csv: one row per four-second segment. Each segment inherits the source work's traditional genre, and source-audio start/end times provide traceability. sampling_method distinguishes fixed-position samples from manual replacements. The final analysis includes only segments whose manual_review_status is keep.\n"
        "segment_overlap_audit.csv: checks adjacent segments within a recording for overlap using the final start/end times.\n"
        "replacement_audit.csv: records the original candidate time, final time, and review reason for manually replaced segments.\n"
        "features_A_mfcc.csv: per-segment means and standard deviations for 13 MFCCs, for 26 dimensions.\n"
        "features_B_spectral.csv: representation A plus means and standard deviations for six spectral-shape descriptors, for 38 dimensions; spectral flatness is stored in dB.\n"
        "features_C_loudness.csv: representation B plus mean and standard deviation of log-RMS/dBFS, for 40 dimensions; loudness is a proxy, not an ITU-R BS.1770 measurement.\n"
        "pca_scores.csv: PCA coordinates for every segment under representations A, B, and C.\n"
        "retrieval_results.csv: Top-5 retrieval results and metadata for every query segment.\n"
        "retrieval_metrics.csv: retrieval metrics, exact random baselines, and recording-grouped bootstrap intervals for P@5.\n"
        "retrieval_top5_stability.csv: pairwise Top-5 candidate-recording membership overlap among A, B, and C for the main retrieval task, with recording-grouped 95% intervals; this does not measure within-list ranking.\n"
        "label_provenance.csv: source, publication correspondence, verification status, and inheritance notes for the work-level genre labels of all 54 recordings.\n"
        "metadata_completeness.csv: known, unknown, and pending counts for key metadata fields.\n"
        "aggregation_sensitivity_summary.csv: macro P@5, 95% confidence intervals, and random-baseline differences for three candidate-recording aggregation methods.\n"
        "sensitivity_low_energy.csv: refits the scaling and recomputes segment-to-recording-prototype P@5 after excluding segments below -40 dBFS.\n"
        "sensitivity_overlap.csv: recomputes within-recording consistency and segment-to-recording-prototype P@5 after excluding the recording with overlapping windows.\n"
        "manual_review_note, quality flags, and unknown fields must not be inferred from machine-generated results.\n", encoding="utf-8")

    pca_rows: List[Dict] = []
    pca_summary: List[Dict] = []
    pca_panels: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    feature_order = {k: [f for f in feature_fields[k] if f not in {"segment_id", "recording_id", "form", "title", "performer", "segment_rms_dbfs"}] for k in ["A", "B", "C"]}
    feature_arrays: Dict[str, np.ndarray] = {}
    segment_ids = [r["segment_id"] for r in feature_rows["A"]]
    record_ids = [r["recording_id"] for r in feature_rows["A"]]
    forms = [r["form"] for r in feature_rows["A"]]
    for key in ["A", "B", "C"]:
        arr = np.array([[float(r[f]) for f in feature_order[key]] for r in feature_rows[key]], dtype=np.float64)
        scaled, med, scale = robust_scale(arr)
        feature_arrays[key] = scaled
        scores, explained = pca(scaled, 5)
        for i, sid in enumerate(segment_ids):
            row = {"segment_id": sid, "recording_id": record_ids[i], "form": forms[i], "representation": key}
            for j in range(5):
                row[f"PC{j + 1}"] = float(scores[i, j])
            pca_rows.append(row)
        for j in range(5):
            pca_summary.append({
                "representation": key, "component": f"PC{j + 1}",
                "explained_variance_ratio": float(explained[j]),
                "cumulative_explained_variance_ratio": float(np.sum(explained[:j + 1])),
            })
        # Save the scaling parameters for exact reruns.
        np.savetxt(OUT_ROOT / f"scaler_{key}_median.csv", med[None, :], delimiter=",", header=",".join(feature_order[key]), comments="")
        np.savetxt(OUT_ROOT / f"scaler_{key}_iqr.csv", scale[None, :], delimiter=",", header=",".join(feature_order[key]), comments="")
        pca_panels[key] = (scores, explained)
        plot_pca(scores, forms, record_ids, f"Tu Pipa sound space: representation {key}", OUT_ROOT / f"pca_{key}.png", explained)
    write_csv(OUT_ROOT / "pca_scores.csv", pca_rows, list(pca_rows[0].keys()))
    write_csv(OUT_ROOT / "pca_summary.csv", pca_summary, list(pca_summary[0].keys()))
    plot_pca_combined(pca_panels, forms, OUT_ROOT / "figure_pca_abc.png")
    plot_corpus_workflow(OUT_ROOT / "figure_corpus_workflow.png")

    # Recording-level prototypes for cross-recording retrieval.
    unique_records = [r["recording_id"] for r in recordings]
    rec_index = {rid: i for i, rid in enumerate(unique_records)}
    proto = {}
    for key, arr in feature_arrays.items():
        p = np.zeros((len(unique_records), arr.shape[1]))
        for rid in unique_records:
            inds = [i for i, rr in enumerate(record_ids) if rr == rid]
            p[rec_index[rid]] = np.mean(arr[inds], axis=0)
        proto[key] = p

    rec_meta = {r["recording_id"]: r for r in recordings}
    retrieval_rows: List[Dict] = []
    metrics_rows: List[Dict] = []
    query_metric_rows: List[Dict] = []
    case_queries: List[Dict] = []
    for key in ["A", "B", "C"]:
        arr = feature_arrays[key]
        p = proto[key]
        per_query: List[Dict] = []
        # Within-recording segment consistency: query against all other clips.
        for qi, sid in enumerate(segment_ids):
            distances = euclidean(arr[qi], arr)
            distances[qi] = np.inf
            order = ranked_indices(distances, segment_ids)
            top = [j for j in order if j != qi][:5]
            rel = [record_ids[j] == record_ids[qi] for j in top]
            per_query.append({
                "representation": key, "task": "within_recording_segments",
                "query_segment_id": sid, "query_recording_id": record_ids[qi], "query_form": forms[qi],
                "p_at_5": float(np.mean(rel)), "hit_rate_at_1": float(rel[0]),
                "hit_rate_at_5": float(any(rel)), "mrr": mrr(rel), "map_at_5": ap_at_k(rel, 5, 4),
            })
            for rank, j in enumerate(top, 1):
                query_meta = rec_meta[record_ids[qi]]
                result_meta = rec_meta[record_ids[j]]
                retrieval_rows.append({
                    "representation": key, "task": "within_recording_segments", "query_segment_id": sid,
                    "query_recording_id": record_ids[qi], "query_form": forms[qi],
                    "query_title": query_meta["title"], "query_performer": query_meta["performer"], "rank": rank,
                    "result_segment_id": segment_ids[j], "result_recording_id": record_ids[j],
                    "result_form": forms[j], "result_title": result_meta["title"],
                    "result_performer": result_meta["performer"],
                    "distance": float(distances[j]), "same_recording": record_ids[j] == record_ids[qi],
                    "same_form": forms[j] == forms[qi], "same_title": result_meta["title"] == query_meta["title"],
                    "performer_overlap": bool(performer_tokens(result_meta["performer"]) & performer_tokens(query_meta["performer"])),
                })
        for task in ["within_recording_segments"]:
            vals = [q["p_at_5"] for q in per_query if q["task"] == task]
            group = [q["query_recording_id"] for q in per_query if q["task"] == task]
            lo, hi = bootstrap_ci(vals, group)
            metrics_rows.append({
                "representation": key, "task": task, "query_count": len(vals),
                "p_at_5": float(np.mean(vals)), "hit_rate_at_1": float(np.mean([q["hit_rate_at_1"] for q in per_query])),
                "hit_rate_at_5": float(np.mean([q["hit_rate_at_5"] for q in per_query])),
                "mrr": float(np.mean([q["mrr"] for q in per_query])), "map_at_5": float(np.mean([q["map_at_5"] for q in per_query])),
                "random_p_at_5_baseline": exact_random_metrics(269, 4)["p_at_5"],
                "random_hit_at_1_baseline": exact_random_metrics(269, 4)["hit_rate_at_1"],
                "random_hit_at_5_baseline": exact_random_metrics(269, 4)["hit_rate_at_5"],
                "random_mrr_at_5_baseline": exact_random_metrics(269, 4)["mrr_at_5"],
                "random_map_at_5_baseline": exact_random_metrics(269, 4)["map_at_5"],
                "p_at_5_baseline_difference": float(np.mean(vals) - exact_random_metrics(269, 4)["p_at_5"]),
                "p_at_5_ci_low": lo, "p_at_5_ci_high": hi,
                "notes": "Bootstrap resamples query recordings and retains all five queries from each sampled recording; rankings and the candidate corpus remain fixed.",
            })
        query_metric_rows.extend(per_query)

        # The primary cross-recording task compares a 4-second query with the
        # mean feature prototype of each other recording. Minimum and mean
        # query-to-candidate-clip distances are retained as aggregation controls.
        cross_queries_by_task: Dict[str, List[Dict]] = {}
        for task, aggregation in [
            ("cross_recording_min_segment", "min"),
            ("cross_recording_mean_segment_distance", "mean"),
            ("cross_recording_prototype", "prototype"),
        ]:
            cross_queries: List[Dict] = []
            for qi, sid in enumerate(segment_ids):
                own_rid = record_ids[qi]
                candidate_scores: List[Tuple[float, str, str]] = []
                for rid in unique_records:
                    if rid == own_rid:
                        continue
                    candidate_indices = [i for i, rr in enumerate(record_ids) if rr == rid]
                    if aggregation == "prototype":
                        score = float(euclidean(arr[qi], p[[rec_index[rid]]])[0])
                        match_segment = "recording_prototype"
                    else:
                        clip_distances = euclidean(arr[qi], arr[candidate_indices])
                        local_order = ranked_indices(clip_distances, [segment_ids[i] for i in candidate_indices])
                        best_local = int(local_order[0])
                        match_segment = segment_ids[candidate_indices[best_local]]
                        score = float(np.min(clip_distances) if aggregation == "min" else np.mean(clip_distances))
                    candidate_scores.append((score, rid, match_segment))
                candidate_scores.sort(key=lambda item: (item[0], item[1]))
                top = candidate_scores[:5]
                rel = [rec_meta[rid]["form"] == forms[qi] for _, rid, _ in top]
                n_relevant = 44 if forms[qi] == "Hua Diao" else 8
                q = {
                    "representation": key, "task": task, "query_segment_id": sid,
                    "query_recording_id": own_rid, "query_form": forms[qi],
                    "p_at_5": float(np.mean(rel)), "hit_rate_at_1": float(rel[0]), "hit_rate_at_5": float(any(rel)),
                    "mrr": mrr(rel), "map_at_5": ap_at_k(rel, 5, n_relevant),
                }
                cross_queries.append(q)
                for rank, (score, rid, match_segment) in enumerate(top, 1):
                    query_meta = rec_meta[own_rid]
                    result_meta = rec_meta[rid]
                    retrieval_rows.append({
                        "representation": key, "task": task, "query_segment_id": sid,
                        "query_recording_id": own_rid, "query_form": forms[qi],
                        "query_title": query_meta["title"], "query_performer": query_meta["performer"], "rank": rank,
                        "result_segment_id": match_segment, "result_recording_id": rid,
                        "result_form": result_meta["form"], "result_title": result_meta["title"],
                        "result_performer": result_meta["performer"], "distance": score,
                        "same_recording": False, "same_form": result_meta["form"] == forms[qi],
                        "same_title": result_meta["title"] == query_meta["title"],
                        "performer_overlap": bool(performer_tokens(result_meta["performer"]) & performer_tokens(query_meta["performer"])),
                    })
            cross_queries_by_task[task] = cross_queries
            for form in [*GENRES, "macro"]:
                subset = cross_queries if form == "macro" else [q for q in cross_queries if q["query_form"] == form]
                vals = [q["p_at_5"] for q in subset]
                groups = [q["query_recording_id"] for q in subset]
                if form == "macro":
                    category_means = [float(np.mean([q["p_at_5"] for q in cross_queries if q["query_form"] == f])) for f in GENRES]
                    estimate = float(np.mean(category_means))
                    aggregate_metrics = {
                        field: float(np.mean([
                            np.mean([q[field] for q in cross_queries if q["query_form"] == f])
                            for f in GENRES
                        ]))
                        for field in ["hit_rate_at_1", "hit_rate_at_5", "mrr", "map_at_5"]
                    }
                    baseline_metrics = {
                        name: float(np.mean([exact_random_metrics(53, 44 if f == GENRE_HUA else 8)[name] for f in GENRES]))
                        for name in ["p_at_5", "hit_rate_at_1", "hit_rate_at_5", "mrr_at_5", "map_at_5"]
                    }
                    lo, hi = bootstrap_macro_ci(vals, groups, [q["query_form"] for q in subset])
                else:
                    estimate = float(np.mean(vals))
                    aggregate_metrics = {
                        field: float(np.mean([q[field] for q in subset]))
                        for field in ["hit_rate_at_1", "hit_rate_at_5", "mrr", "map_at_5"]
                    }
                    baseline_metrics = exact_random_metrics(53, 44 if form == GENRE_HUA else 8)
                    lo, hi = bootstrap_ci(vals, groups)
                metrics_rows.append({
                    "representation": key, "task": task, "query_form": form,
                    "query_count": len(vals), "p_at_5": estimate,
                    **aggregate_metrics,
                    "random_p_at_5_baseline": baseline_metrics["p_at_5"],
                    "random_hit_at_1_baseline": baseline_metrics["hit_rate_at_1"],
                    "random_hit_at_5_baseline": baseline_metrics["hit_rate_at_5"],
                    "random_mrr_at_5_baseline": baseline_metrics["mrr_at_5"],
                    "random_map_at_5_baseline": baseline_metrics["map_at_5"],
                    "p_at_5_baseline_difference": estimate - baseline_metrics["p_at_5"],
                    "p_at_5_ci_low": lo, "p_at_5_ci_high": hi,
                    "notes": "Query source recording excluded; macro averages the two category-specific metrics equally. Rankings and candidate corpus remain fixed for bootstrap.",
                })
            query_metric_rows.extend(cross_queries)
        # Cases are selected from the primary recording-prototype task so that
        # the previously completed human listening notes remain traceable.
        cross_queries = cross_queries_by_task["cross_recording_prototype"]
        if key == "B":
            ranked = sorted(cross_queries, key=lambda q: (q["p_at_5"], q["mrr"]), reverse=True)
            if ranked:
                case_queries.append({**ranked[0], "case_type": "successful_same_form", "case_selection": "highest P@5 among all B queries"})
            mixed = sorted([q for q in cross_queries if 0 < q["p_at_5"] < 1], key=lambda q: abs(q["p_at_5"] - 0.4))
            if mixed:
                case_queries.append({**mixed[0], "case_type": "partial_intersection", "case_selection": "mixed top-5 form composition"})
            failed = [q for q in cross_queries if q["p_at_5"] == 0]
            if failed:
                case_queries.append({**failed[0], "case_type": "divergent_or_failed", "case_selection": "no same-form result in top-5"})
    retrieval_fields = [
        "representation", "task", "query_segment_id", "query_recording_id", "query_form",
        "query_title", "query_performer", "rank", "result_segment_id", "result_recording_id",
        "result_form", "result_title", "result_performer", "distance", "same_recording",
        "same_form", "same_title", "performer_overlap",
    ]
    metric_fields = [
        "representation", "task", "query_form", "query_count", "p_at_5", "hit_rate_at_1",
        "hit_rate_at_5", "mrr", "map_at_5", "random_p_at_5_baseline",
        "random_hit_at_1_baseline", "random_hit_at_5_baseline", "random_mrr_at_5_baseline",
        "random_map_at_5_baseline", "p_at_5_baseline_difference",
        "p_at_5_ci_low", "p_at_5_ci_high", "notes",
    ]
    write_csv(OUT_ROOT / "retrieval_results.csv", retrieval_rows, retrieval_fields)
    write_csv(OUT_ROOT / "retrieval_metrics.csv", metrics_rows, metric_fields)
    write_csv(OUT_ROOT / "retrieval_query_metrics.csv", query_metric_rows, list(query_metric_rows[0].keys()))
    aggregation_labels = {
        "cross_recording_min_segment": "Minimum segment distance",
        "cross_recording_mean_segment_distance": "Mean segment distance",
        "cross_recording_prototype": "Recording prototype (main analysis)",
    }
    aggregation_summary_rows = []
    for row in metrics_rows:
        if row.get("task") not in aggregation_labels or row.get("query_form") != "macro":
            continue
        aggregation_summary_rows.append({
            "aggregation": aggregation_labels[row["task"]],
            "task": row["task"],
            "representation": row["representation"],
            "macro_p_at_5": row["p_at_5"],
            "ci_low": row["p_at_5_ci_low"],
            "ci_high": row["p_at_5_ci_high"],
            "random_macro_baseline": row["random_p_at_5_baseline"],
            "baseline_difference": row["p_at_5_baseline_difference"],
        })
    write_csv(
        OUT_ROOT / "aggregation_sensitivity_summary.csv",
        aggregation_summary_rows,
        ["aggregation", "task", "representation", "macro_p_at_5", "ci_low", "ci_high", "random_macro_baseline", "baseline_difference"],
    )

    # Paired descriptive comparisons for the main cross-recording task. The
    # intervals resample query recordings and do not constitute a claim that
    # one representation is universally superior outside this fixed corpus.
    main_query_rows = [row for row in query_metric_rows if row["task"] == "cross_recording_prototype"]
    query_lookup = {
        (row["representation"], row["query_segment_id"]): row
        for row in main_query_rows
    }
    difference_rows: List[Dict] = []
    for left, right in [("A", "B"), ("A", "C"), ("B", "C")]:
        paired = []
        for sid, rid, form in zip(segment_ids, record_ids, forms):
            diff = float(query_lookup[(left, sid)]["p_at_5"] - query_lookup[(right, sid)]["p_at_5"])
            paired.append((diff, rid, form))
        for form in [*GENRES, "macro"]:
            subset = paired if form == "macro" else [row for row in paired if row[2] == form]
            values = [row[0] for row in subset]
            if form == "macro":
                estimate = float(np.mean([
                    np.mean([row[0] for row in paired if row[2] == f]) for f in GENRES
                ]))
                lo, hi = bootstrap_macro_ci(values, [row[1] for row in subset], [row[2] for row in subset])
            else:
                estimate = float(np.mean(values))
                lo, hi = bootstrap_ci(values, [row[1] for row in subset])
            difference_rows.append({
                "task": "cross_recording_prototype", "comparison": f"{left}-{right}",
                "query_form": form, "p_at_5_difference": estimate,
                "ci_low": lo, "ci_high": hi, "interval_contains_zero": lo <= 0.0 <= hi,
                "notes": "Paired by query segment; bootstrap resamples query recordings with fixed rankings and candidate corpus.",
            })
    write_csv(
        OUT_ROOT / "retrieval_representation_differences.csv", difference_rows,
        list(difference_rows[0].keys()),
    )

    # Cross-representation Top-5 neighbor-set stability for the primary
    # cross-recording prototype task. This measures membership overlap only;
    # it is not a rank-correlation or a classification score.
    main_rows = [
        row for row in retrieval_rows
        if row["task"] == "cross_recording_prototype"
    ]
    top_sets: Dict[Tuple[str, str], set[str]] = {}
    query_forms: Dict[str, str] = {}
    query_recordings: Dict[str, str] = {}
    for key in ["A", "B", "C"]:
        grouped: Dict[str, List[Dict]] = {}
        for row in main_rows:
            if row["representation"] == key:
                grouped.setdefault(row["query_segment_id"], []).append(row)
                query_forms[row["query_segment_id"]] = row["query_form"]
                query_recordings[row["query_segment_id"]] = row["query_recording_id"]
        for sid, rows in grouped.items():
            top_sets[(key, sid)] = {
                row["result_recording_id"]
                for row in rows
                if int(row["rank"]) <= 5
            }
    stability_rows: List[Dict] = []
    for left, right in [("A", "B"), ("A", "C"), ("B", "C")]:
        paired = []
        for sid in segment_ids:
            left_set = top_sets[(left, sid)]
            right_set = top_sets[(right, sid)]
            assert len(left_set) == 5 and len(right_set) == 5
            overlap = len(left_set & right_set) / 5.0
            paired.append((overlap, query_recordings[sid], query_forms[sid]))
        for form in [*GENRES, "macro"]:
            subset = paired if form == "macro" else [row for row in paired if row[2] == form]
            values = [row[0] for row in subset]
            groups = [row[1] for row in subset]
            if form == "macro":
                estimate = float(np.mean([
                    np.mean([row[0] for row in paired if row[2] == f])
                    for f in GENRES
                ]))
                lo, hi = bootstrap_macro_ci(values, groups, [row[2] for row in subset])
            else:
                estimate = float(np.mean(values))
                lo, hi = bootstrap_ci(values, groups)
            stability_rows.append({
                "task": "cross_recording_prototype",
                "comparison": f"{left}-{right}",
                "query_form": form,
                "query_count": len(values),
                "mean_top5_overlap": estimate,
                "median_top5_overlap": float(np.median(values)),
                "ci_low": lo,
                "ci_high": hi,
                "overlap_definition": "|Top5_left ∩ Top5_right| / 5; recording IDs as candidates",
                "notes": "Membership overlap only; bootstrap resamples query recordings and retains all five queries from each sampled recording.",
            })
    write_csv(
        OUT_ROOT / "retrieval_top5_stability.csv", stability_rows,
        ["task", "comparison", "query_form", "query_count", "mean_top5_overlap", "median_top5_overlap", "ci_low", "ci_high", "overlap_definition", "notes"],
    )

    case_rows: List[Dict] = []
    for case in case_queries:
        query_meta = rec_meta[case["query_recording_id"]]
        matches = [r for r in retrieval_rows if r["representation"] == case["representation"] and r["task"] == "cross_recording_prototype" and r["query_segment_id"] == case["query_segment_id"]]
        for result in sorted(matches, key=lambda r: int(r["rank"])):
            case_rows.append({
                "case_type": case["case_type"], "selection_stage": "post_hoc_explanatory",
                "case_selection": case["case_selection"],
                "representation": case["representation"], "query_segment_id": case["query_segment_id"],
                "query_recording_id": case["query_recording_id"], "query_form": case["query_form"],
                "query_title": query_meta["title"], "query_performer": query_meta["performer"],
                "query_p_at_5": case["p_at_5"], "rank": result["rank"],
                "result_recording_id": result["result_recording_id"], "result_form": result["result_form"],
                "result_title": result["result_title"], "result_performer": result["result_performer"],
                "distance": result["distance"], "same_form": result["same_form"],
                "manual_interpretation": manual_case_interpretations.get(
                    (case["case_type"], case["query_segment_id"], result["result_recording_id"]),
                    "New neighbor after the final recomputation; not separately reviewed by listening",
                ),
            })
    write_csv(OUT_ROOT / "retrieval_cases.csv", case_rows, list(case_rows[0].keys()) if case_rows else ["representation", "case_type"])

    # Low-energy sensitivity check: remove the two segment-level clips below
    # -40 dBFS, refit robust scaling, and repeat the primary prototype task.
    segment_quality = {row["segment_id"]: row["segment_quality_flag"] for row in segments}
    sensitivity_rows: List[Dict] = []
    for key in ["A", "B", "C"]:
        kept_rows = [
            row for row in feature_rows[key]
            if segment_quality.get(row["segment_id"]) != "review_low_energy"
        ]
        raw = np.array([[float(row[f]) for f in feature_order[key]] for row in kept_rows], dtype=np.float64)
        scaled, _, _ = robust_scale(raw)
        kept_ids = [row["segment_id"] for row in kept_rows]
        kept_record_ids = [row["recording_id"] for row in kept_rows]
        kept_forms = [row["form"] for row in kept_rows]
        sensitivity_records = sorted(set(kept_record_ids))
        sensitivity_rec_index = {rid: i for i, rid in enumerate(sensitivity_records)}
        sensitivity_proto = np.stack([
            np.mean(scaled[[i for i, rid2 in enumerate(kept_record_ids) if rid2 == rid]], axis=0)
            for rid in sensitivity_records
        ])
        for form in GENRES:
            p_at_5_values = []
            for i, sid in enumerate(kept_ids):
                if kept_forms[i] != form:
                    continue
                distances = euclidean(scaled[i], sensitivity_proto)
                distances[sensitivity_rec_index[kept_record_ids[i]]] = np.inf
                order = ranked_indices(distances, sensitivity_records)
                top = [(float(distances[j]), sensitivity_records[j]) for j in order[:5]]
                p_at_5_values.append(float(np.mean([
                    rec_meta[rid]["form"] == form for _, rid in top
                ])))
            full_row = next(
                row for row in metrics_rows
                if row.get("representation") == key
                and row.get("task") == "cross_recording_prototype"
                and row.get("query_form") == form
            )
            sensitivity_p_at_5 = float(np.mean(p_at_5_values))
            sensitivity_rows.append({
                "representation": key,
                "query_form": form,
                "full_query_count": full_row["query_count"],
                "sensitivity_query_count": len(p_at_5_values),
                "excluded_low_energy_segments": len(feature_rows[key]) - len(kept_rows),
                "full_p_at_5": full_row["p_at_5"],
                "exclude_low_energy_p_at_5": sensitivity_p_at_5,
                "difference": sensitivity_p_at_5 - float(full_row["p_at_5"]),
            })
    write_csv(
        OUT_ROOT / "sensitivity_low_energy.csv",
        sensitivity_rows,
        list(sensitivity_rows[0].keys()),
    )

    # Overlap sensitivity: exclude every recording whose final reviewed
    # windows overlap and repeat within-recording and main cross-recording P@5.
    # This quantifies rather than assumes the influence of shared audio.
    overlap_recordings = sorted({row["recording_id"] for row in overlap_rows})
    overlap_sensitivity_rows: List[Dict] = []
    for key in ["A", "B", "C"]:
        kept_indices = [i for i, rid in enumerate(record_ids) if rid not in overlap_recordings]
        arr = feature_arrays[key][kept_indices]
        kept_ids = [segment_ids[i] for i in kept_indices]
        kept_rids = [record_ids[i] for i in kept_indices]
        kept_forms = [forms[i] for i in kept_indices]
        # Within-recording task.
        within_values = []
        for qi in range(len(kept_ids)):
            distances = euclidean(arr[qi], arr)
            distances[qi] = np.inf
            order = ranked_indices(distances, kept_ids)[:5]
            within_values.append(float(np.mean([kept_rids[j] == kept_rids[qi] for j in order])))
        full_within = next(row for row in metrics_rows if row["representation"] == key and row["task"] == "within_recording_segments")
        overlap_sensitivity_rows.append({
            "representation": key, "task": "within_recording_segments", "query_form": "all",
            "exclusion_unit": "entire_recording",
            "excluded_recordings": ";".join(overlap_recordings), "full_p_at_5": full_within["p_at_5"],
            "full_query_count": len(segment_ids), "sensitivity_query_count": len(kept_ids),
            "candidate_recordings_after": len(set(kept_rids)),
            "exclude_overlap_recording_p_at_5": float(np.mean(within_values)),
            "difference": float(np.mean(within_values)) - float(full_within["p_at_5"]),
        })
        # Primary cross-recording task: recording prototypes.
        sensitivity_records = sorted(set(kept_rids))
        sensitivity_rec_index = {rid: i for i, rid in enumerate(sensitivity_records)}
        sensitivity_proto = np.stack([
            np.mean(arr[[j for j, rr in enumerate(kept_rids) if rr == rid]], axis=0)
            for rid in sensitivity_records
        ])
        for form in GENRES:
            values = []
            for qi in range(len(kept_ids)):
                if kept_forms[qi] != form:
                    continue
                distances = euclidean(arr[qi], sensitivity_proto)
                distances[sensitivity_rec_index[kept_rids[qi]]] = np.inf
                order = ranked_indices(distances, sensitivity_records)[:5]
                values.append(float(np.mean([rec_meta[sensitivity_records[j]]["form"] == form for j in order])))
            full_cross = next(
                row for row in metrics_rows
                if row["representation"] == key and row["task"] == "cross_recording_prototype" and row["query_form"] == form
            )
            overlap_sensitivity_rows.append({
                "representation": key, "task": "cross_recording_prototype", "query_form": form,
                "exclusion_unit": "entire_recording",
                "excluded_recordings": ";".join(overlap_recordings), "full_p_at_5": full_cross["p_at_5"],
                "full_query_count": sum(current_form == form for current_form in forms),
                "sensitivity_query_count": len(values),
                "candidate_recordings_after": len(sensitivity_records),
                "exclude_overlap_recording_p_at_5": float(np.mean(values)),
                "difference": float(np.mean(values)) - float(full_cross["p_at_5"]),
            })
    write_csv(
        OUT_ROOT / "sensitivity_overlap.csv", overlap_sensitivity_rows,
        list(overlap_sensitivity_rows[0].keys()),
    )

    summary = [
        "Summary of the Tu Pipa corpus analysis",
        f"Generated on: {datetime.now(timezone.utc).date().isoformat()}",
        f"Core recordings: {len(recordings)}; candidate segments: {len(segments)}; segments retained after manual review: {len(accepted_ids)}.",
        f"Manually replaced segment positions: {sum(row.get('sampling_method') == 'manual_replacement' for row in segments)}; five four-second segments were retained for every recording.",
        "Segments are sampled from recordings, and retrieval evaluation is grouped by source recording. The 270 segments must not be treated as 270 independent field recordings.",
        "",
        "PCA summary:",
    ]
    for row in pca_summary:
        if row["component"] in {"PC1", "PC2"}:
            summary.append(f"Representation {row['representation']}, {row['component']} explained variance = {row['explained_variance_ratio']:.4f}; cumulative = {row['cumulative_explained_variance_ratio']:.4f}")
    summary += ["", "Retrieval summary (see retrieval_metrics.csv):"]
    for row in metrics_rows:
        if row.get("task") == "cross_recording_prototype" and row.get("query_form") in set(GENRES):
            summary.append(f"Representation {row['representation']}, {row['query_form']}: P@5 = {row['p_at_5']:.4f}; random baseline = {row.get('random_p_at_5_baseline', float('nan')):.4f}; difference = {row.get('p_at_5_baseline_difference', float('nan')):.4f}")
    summary += ["", "Top-5 neighbor-set stability for the primary retrieval task (macro-average):"]
    for row in stability_rows:
        if row["query_form"] == "macro":
            summary.append(f"{row['comparison']} mean overlap = {row['mean_top5_overlap']:.4f}; 95% CI = {row['ci_low']:.4f}-{row['ci_high']:.4f}; median = {row['median_top5_overlap']:.4f}")
    summary += ["", "Quality note: `low_level` and `review_low_energy` are review flags, not automatic evidence of an error; interpret them alongside listening checks and sensitivity analyses."]
    (OUT_ROOT / "analysis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
