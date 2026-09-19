"""
Build VoiceGuard deployable Prosody Fusion V1.

Usage:
    python build_voiceguard_prosody_fusion_v1.py

Protocol:
1. Extract prosody features from TRAIN only.
2. Fit StandardScaler + LogisticRegression on TRAIN only.
3. Run frozen V2 + prosody on VALIDATION.
4. Select alpha over a fixed grid on VALIDATION only.
5. Save scorer + fusion configuration.
6. Do not read TEST.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from voiceguard_prosody_fusion_v1 import (
    FEATURE_NAMES, ProsodyScorer, extract_prosody_features, save_scorer
)
from inference import VoiceGuardInference

MASTER = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
V2 = ROOT / "experiments" / "reverb_v2" / "models" / "best_model.pt"
OUT = ROOT / "experiments" / "reverb_v2" / "prosody_fusion_v1"
SCORER = OUT / "prosody_scorer_v1.pkl"
CONFIG = OUT / "PROSODY_FUSION_CONFIG_V1.json"

def rows_for(split):
    with MASTER.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            if row["split"] != split:
                continue
            p = Path(row["processed_audio_path"])
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                rows.append((row, p))
        return rows

def label(row):
    v = row["label"].strip().lower()
    if v in {"spoof", "fake", "1"}:
        return 1
    if v in {"bonafide", "real", "genuine", "0"}:
        return 0
    raise ValueError(f"Unknown label: {row['label']}")

def extract(rows):
    X, y = [], []
    for i, (row, path) in enumerate(rows, 1):
        wav, sr = librosa.load(str(path), sr=16000, mono=True)
        f = extract_prosody_features(wav, sr)
        X.append([f[n] for n in FEATURE_NAMES])
        y.append(label(row))
        if i % 250 == 0:
            print(f"  extracted {i}/{len(rows)}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

def main():
    print("=" * 72)
    print("VOICEGUARD — DEPLOYABLE PROSODY FUSION V1 BUILDER")
    print("=" * 72)

    train = rows_for("train")
    val = rows_for("validation")
    print("Train rows:", len(train))
    print("Validation rows:", len(val))
    print("Protected test rows: NOT READ")

    Xtr, ytr = extract(train)
    Xva, yva = extract(val)

    scaler = StandardScaler()
    Xtrz = scaler.fit_transform(Xtr)
    Xvaz = scaler.transform(Xva)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(Xtrz, ytr)
    scorer = ProsodyScorer(scaler=scaler, classifier=clf, feature_names=FEATURE_NAMES)

    p_pros = clf.predict_proba(Xvaz)[:, 1]

    # Frozen V2 inference on validation only for fusion selection.
    v2_engine = VoiceGuardInference(checkpoint_path=V2)
    p_v2 = np.array(
        [v2_engine.predict(path)["spoof_probability"] for _, path in val],
        dtype=np.float64,
    )

    best = None
    for alpha in np.linspace(0.70, 0.99, 30):
        fused = alpha * p_v2 + (1.0 - alpha) * p_pros
        pred = (fused >= 0.25).astype(int)
        f1 = f1_score(yva, pred)
        bal = balanced_accuracy_score(yva, pred)
        candidate = (f1, bal, alpha)
        if best is None or candidate > best:
            best = candidate

    f1, bal, alpha = best
    pros_weight = 1.0 - alpha

    metadata = {
        "artifact": "voiceguard_prosody_scorer_v1",
        "fit_split": "train",
        "fusion_selection_split": "validation",
        "protected_test_read": False,
        "feature_names": FEATURE_NAMES,
        "classifier": "StandardScaler + LogisticRegression",
        "random_state": 42,
        "frozen_v2_threshold": 0.25,
        "primary_weight": float(alpha),
        "prosody_weight": float(pros_weight),
        "validation_f1": float(f1),
        "validation_balanced_accuracy": float(bal),
        "note": "Fusion weights are validation-selected and must be treated as a frozen engineering candidate until independently evaluated.",
    }
    save_scorer(scorer, SCORER, metadata)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nPROSODY SCORER FIT: PASS")
    print("Fusion selection: PASS (validation only)")
    print(f"Primary weight : {alpha:.6f}")
    print(f"Prosody weight : {pros_weight:.6f}")
    print(f"Validation F1  : {f1:.6f}")
    print(f"Validation BAL : {bal:.6f}")
    print("Protected test : NOT READ")
    print("Artifact:", SCORER)
    print("Config  :", CONFIG)

if __name__ == "__main__":
    main()
