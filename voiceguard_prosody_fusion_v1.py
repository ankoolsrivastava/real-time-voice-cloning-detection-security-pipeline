"""
VoiceGuard Prosody Fusion V1

Train-on-TRAIN / select-on-VALIDATION / freeze artifact.

This creates a deployable secondary prosody scorer:
    audio -> deterministic prosody features -> scaler + logistic regression
    -> spoof_probability

It must never read the protected TEST split during fitting or selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import pickle

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "duration_sec",
    "rms_mean",
    "rms_std",
    "rms_dynamic_range",
    "pause_total_sec",
    "pause_mean_sec",
    "voiced_ratio",
    "voiced_transition_rate",
]

@dataclass
class ProsodyScorer:
    scaler: StandardScaler
    classifier: LogisticRegression
    feature_names: List[str]

    def predict_proba(self, features: Dict[str, float]) -> float:
        x = np.array([[features[n] for n in self.feature_names]], dtype=np.float32)
        z = self.scaler.transform(x)
        return float(self.classifier.predict_proba(z)[0, 1])

def extract_prosody_features(waveform: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    y = np.asarray(waveform, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Empty waveform")

    frame_length = 400
    hop = 160
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop, center=True
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop, center=True
    )[0]

    # Deterministic activity/voicing proxy. This is a behavioral/prosodic
    # feature extractor, not a phonetic voicing detector.
    threshold = max(float(np.percentile(rms, 35)) * 0.8, 1e-5)
    active = rms > threshold

    transitions = np.sum(active[1:] != active[:-1]) if len(active) > 1 else 0
    transition_rate = float(transitions / max(len(active) - 1, 1))

    # Approximate pauses from inactive frames.
    inactive_frames = ~active
    pause_frame_count = int(np.sum(inactive_frames))
    pause_total = pause_frame_count * hop / sr

    runs = []
    start = None
    for i, flag in enumerate(inactive_frames):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append(i - start)
            start = None
    if start is not None:
        runs.append(len(inactive_frames) - start)

    pause_lengths = [r * hop / sr for r in runs if r > 0]
    pause_mean = float(np.mean(pause_lengths)) if pause_lengths else 0.0

    positive_rms = rms[rms > 1e-8]
    if positive_rms.size:
        dynamic_range = float(
            20.0 * np.log10(np.max(positive_rms) / np.min(positive_rms))
        )
    else:
        dynamic_range = 0.0

    return {
        "duration_sec": float(len(y) / sr),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_dynamic_range": dynamic_range,
        "pause_total_sec": pause_total,
        "pause_mean_sec": pause_mean,
        "voiced_ratio": float(np.mean(active)),
        "voiced_transition_rate": transition_rate,
        "zcr_mean": float(np.mean(zcr)),  # diagnostic, not used by scorer
    }

def save_scorer(scorer: ProsodyScorer, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(scorer, f)
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def load_scorer(path: Path) -> ProsodyScorer:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, ProsodyScorer):
        raise TypeError("Invalid VoiceGuard prosody scorer artifact")
    return obj
