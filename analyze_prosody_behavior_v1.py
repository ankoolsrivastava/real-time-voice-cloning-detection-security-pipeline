from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(r"D:\VoiceGaurd")
MANIFEST = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
OUT = ROOT / "experiments" / "reverb_v2" / "prosody_analysis"
OUT.mkdir(parents=True, exist_ok=True)

ROWS_CSV = OUT / "PROSODY_FEATURES_SAMPLE.csv"
REPORT_JSON = OUT / "PROSODY_ANALYSIS_REPORT.json"


def pick_audio_column(fields):
    for name in ("audio_path", "filepath", "file_path", "path",
                 "processed_path", "audio_file", "file"):
        if name in fields:
            return name
    for name in fields:
        if "audio" in name.lower() and "path" in name.lower():
            return name
    raise RuntimeError(f"No audio path column: {fields}")


def safe_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": None, "std": None, "median": None,
                "min": None, "max": None}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def extract(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) == 0:
        raise ValueError("empty audio")

    # Peak-normalized only for feature stability; source audio is untouched.
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    duration = len(y) / sr

    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)[0]

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=70,
        fmax=350,
        sr=sr,
        frame_length=1024,
        hop_length=160,
    )

    voiced = f0[np.isfinite(f0)]
    voiced_ratio = float(np.mean(np.isfinite(f0))) if len(f0) else 0.0

    # Silence/pausing proxy from frame RMS relative to file RMS.
    rms_floor = max(float(np.median(rms)) * 0.35, 1e-5)
    silent = rms < rms_floor

    # Runs of silent frames, excluding tiny gaps.
    runs = []
    run = 0
    for s in silent:
        if s:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)

    pause_durations = [r * 160 / sr for r in runs if r * 160 / sr >= 0.15]

    # Energy dynamics and spectral centroid movement.
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=400, hop_length=160
    )[0]

    # A simple rate proxy: voiced pitch-cycle events are not reliable enough
    # across languages, so use voiced/unvoiced transitions per second.
    voiced_mask = np.isfinite(f0)
    transitions = int(np.sum(voiced_mask[1:] != voiced_mask[:-1])) if len(voiced_mask) > 1 else 0
    transition_rate = transitions / max(duration, 1e-6)

    return {
        "duration_sec": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_dynamic_range": float(np.percentile(rms, 95) - np.percentile(rms, 5)),
        "zcr_mean": float(np.mean(zcr)),
        "f0_mean_hz": float(np.mean(voiced)) if len(voiced) else None,
        "f0_std_hz": float(np.std(voiced)) if len(voiced) else None,
        "f0_median_hz": float(np.median(voiced)) if len(voiced) else None,
        "f0_range_hz": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)) if len(voiced) else None,
        "voiced_ratio": voiced_ratio,
        "pause_count": len(pause_durations),
        "pause_total_sec": float(sum(pause_durations)),
        "pause_mean_sec": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "voiced_transition_rate": transition_rate,
        "spectral_centroid_mean_hz": float(np.mean(centroid)),
        "spectral_centroid_std_hz": float(np.std(centroid)),
    }


def effect_direction(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    return "spoof_higher" if b > a else "bonafide_higher"


def main():
    print("=" * 72)
    print("VOICEGUARD — PROSODY / BEHAVIORAL ANALYSIS")
    print("=" * 72)

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        audio_col = pick_audio_column(reader.fieldnames or [])

    # Balanced, deterministic diagnostic sample: 250 REAL + 250 SPOOF,
    # with language balancing where possible. This is an analysis sample,
    # not a replacement dataset and not a training split.
    usable = []
    for r in rows:
        raw = str(r.get(audio_col, "")).strip()
        if not raw:
            continue

        # Resolve both absolute paths and paths relative to D:\VoiceGaurd.
        candidates = [Path(raw), ROOT / raw]
        p = next((c for c in candidates if c.exists() and c.is_file()), None)

        # Tolerate a D:\VoiceGaurd path stored with a different slash style.
        if p is None:
            normalized = raw.replace("/", "\\")
            marker = r"D:\VoiceGaurd"
            if normalized.lower().startswith(marker.lower()):
                candidate = ROOT / normalized[len(marker):].lstrip("\\")
                if candidate.exists() and candidate.is_file():
                    p = candidate

        if p is not None:
            usable.append((p, r))

    selected = []
    targets = [
        ("bonafide", "HI", 125),
        ("bonafide", "MR", 125),
        ("spoof", "HI", 125),
        ("spoof", "MR", 125),
    ]

    for label, lang, n in targets:
        count = 0
        for p, r in usable:
            if str(r.get("label", "")).lower() == label and str(r.get("language", "")).upper() == lang:
                selected.append((p, r))
                count += 1
                if count >= n:
                    break

    if len(selected) < 500:
        raise RuntimeError(f"Expected 500 selected files, got {len(selected)}")

    feature_rows = []
    failures = 0

    for i, (p, r) in enumerate(selected, 1):
        try:
            features = extract(str(p))
            feature_rows.append({
                "audio_path": str(p),
                "label": r.get("label", ""),
                "language": r.get("language", ""),
                "condition": r.get("condition", ""),
                "generator": r.get("generator", ""),
                **features,
            })
        except Exception as e:
            failures += 1
            print(f"FAIL {p}: {e}")

        if i % 50 == 0:
            print(f"Processed {i}/{len(selected)}")

    if not feature_rows:
        raise RuntimeError("No feature rows generated")

    with ROWS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    feature_names = [
        "duration_sec", "rms_mean", "rms_std", "rms_dynamic_range",
        "zcr_mean", "f0_mean_hz", "f0_std_hz", "f0_median_hz",
        "f0_range_hz", "voiced_ratio", "pause_count", "pause_total_sec",
        "pause_mean_sec", "voiced_transition_rate",
        "spectral_centroid_mean_hz", "spectral_centroid_std_hz",
    ]

    overall = {}
    for label in ("bonafide", "spoof"):
        subset = [r for r in feature_rows if r["label"].lower() == label]
        overall[label] = {
            f: safe_stats([r[f] for r in subset if r[f] is not None])
            for f in feature_names
        }

    language = {}
    for lang_code, lang_name in (("HI", "hindi"), ("MR", "marathi")):
        language[lang_name] = {}
        for label in ("bonafide", "spoof"):
            subset = [
                r for r in feature_rows
                if r["language"].upper() == lang_code and r["label"].lower() == label
            ]
            language[lang_name][label] = {
                f: safe_stats([r[f] for r in subset if r[f] is not None])
                for f in feature_names
            }

    # Standardized mean difference (Cohen-style d using pooled SD) is used
    # only as a descriptive separation statistic, not as a classifier metric.
    separation = {}
    for f in feature_names:
        b = np.asarray([r[f] for r in feature_rows if r["label"].lower() == "bonafide" and r[f] is not None], dtype=float)
        s = np.asarray([r[f] for r in feature_rows if r["label"].lower() == "spoof" and r[f] is not None], dtype=float)
        if len(b) > 1 and len(s) > 1:
            pooled = math.sqrt(((len(b)-1)*np.var(b, ddof=1) + (len(s)-1)*np.var(s, ddof=1)) / (len(b)+len(s)-2))
            d = float((np.mean(s)-np.mean(b))/pooled) if pooled > 0 else 0.0
            separation[f] = {
                "standardized_mean_difference_spoof_minus_bonafide": d,
                "absolute_effect_size": abs(d),
                "direction": effect_direction(float(np.mean(b)), float(np.mean(s))),
            }

    ranked = sorted(
        separation.items(),
        key=lambda kv: kv[1]["absolute_effect_size"],
        reverse=True,
    )

    report = {
        "analysis_version": "voiceguard_prosody_behavior_v1",
        "manifest": str(MANIFEST),
        "sample_count_requested": 500,
        "sample_count_analyzed": len(feature_rows),
        "failures": failures,
        "selection": {
            "bonafide_HI": 125,
            "bonafide_MR": 125,
            "spoof_HI": 125,
            "spoof_MR": 125,
        },
        "sample_is_diagnostic_only": True,
        "source_data_unchanged": True,
        "features": feature_names,
        "overall_descriptive_statistics": overall,
        "language_descriptive_statistics": language,
        "descriptive_separation": dict(ranked),
        "interpretation_guardrail": (
            "These are descriptive prosodic/acoustic statistics on a diagnostic sample. "
            "Effect size does not establish generalization, causality, or production detection "
            "performance. Prosody is not fused into the frozen V2 classifier by this analysis."
        ),
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("PROSODY ANALYSIS COMPLETE")
    print("=" * 72)
    print("Analyzed :", len(feature_rows))
    print("Failures :", failures)
    print("CSV      :", ROWS_CSV)
    print("Report   :", REPORT_JSON)
    print()
    print("Top descriptive separation features:")
    for name, stats in ranked[:8]:
        print(
            f"  {name:30s} | "
            f"|d|={stats['absolute_effect_size']:.4f} | "
            f"{stats['direction']}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
