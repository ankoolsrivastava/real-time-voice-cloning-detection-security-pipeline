from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import librosa
import torch

ROOT = Path(r"D:\VoiceGaurd")
SCRIPTS = ROOT / "scripts"
MANIFEST = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
CHECKPOINT = ROOT / "experiments" / "reverb_v2" / "models" / "best_model.pt"
PROSODY_CSV = ROOT / "experiments" / "reverb_v2" / "prosody_analysis" / "PROSODY_FEATURES_SAMPLE.csv"
OUT = ROOT / "experiments" / "reverb_v2" / "prosody_incremental"
OUT.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT / "PROSODY_INCREMENTAL_VALIDATION_PREDICTIONS.csv"
REPORT_JSON = OUT / "PROSODY_INCREMENTAL_REPORT.json"

import sys
sys.path.insert(0, str(SCRIPTS))
from audio_preprocessor import VoiceGuardAudioPreprocessor
from feature_extractor import VoiceGuardFeatureExtractor
from model import VoiceGuardModel


FEATURES = [
    "duration_sec", "rms_mean", "rms_std", "rms_dynamic_range",
    "zcr_mean", "f0_mean_hz", "f0_std_hz", "f0_median_hz",
    "f0_range_hz", "voiced_ratio", "pause_count", "pause_total_sec",
    "pause_mean_sec", "voiced_transition_rate",
    "spectral_centroid_mean_hz", "spectral_centroid_std_hz",
]


def resolve(raw):
    p = Path(str(raw))
    if p.exists():
        return p
    p = ROOT / str(raw)
    if p.exists():
        return p
    return None


def auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    order = np.argsort(-scores)
    y = labels[order]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None
    tpr = np.cumsum(y == 1) / pos
    fpr = np.cumsum(y == 0) / neg
    return float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))


def metrics(prob, y, threshold=0.25):
    pred = (np.asarray(prob) >= threshold).astype(int)
    y = np.asarray(y)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    n = len(y)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "balanced_accuracy": ((tp/(tp+fn)) + (tn/(tn+fp))) / 2 if (tp+fn and tn+fp) else 0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "roc_auc": auc(prob, y),
    }


def main():
    print("=" * 72)
    print("VOICEGUARD — PROSODY INCREMENTAL VALIDATION EXPERIMENT")
    print("=" * 72)

    # Validation only. We intentionally do not touch clean test.
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    val = []
    for r in rows:
        if str(r.get("split", "")).lower() != "validation":
            continue
        p = resolve(r.get("processed_audio_path", ""))
        if p is not None and str(r.get("label", "")).lower() in ("bonafide", "spoof"):
            val.append((p, r))

    # Use the already-computed prosody diagnostic features only where the
    # exact validation audio path is present. No test-set inference.
    with PROSODY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        pr = list(csv.DictReader(f))

    prosody_by_path = {str(resolve(r["audio_path"])): r for r in pr if resolve(r["audio_path"])}

    # Since the 500-file diagnostic sample is intentionally small, build
    # prosody features for the full validation set here. This gives a complete,
    # split-correct validation experiment rather than an overlap-only result.
    pre = VoiceGuardAudioPreprocessor(target_sr=16000, max_duration_sec=10.0)

    print("Validation files:", len(val))
    print("Computing validation prosody features...")

    rows_out = []
    failures = 0

    for i, (path, r) in enumerate(val, 1):
        try:
            y, sr = librosa.load(str(path), sr=16000, mono=True)
            if len(y) == 0:
                raise ValueError("empty audio")
            peak = float(np.max(np.abs(y)))
            if peak > 0:
                y = y / peak
            duration = len(y) / sr
            rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)[0]
            f0, _, _ = librosa.pyin(y, fmin=70, fmax=350, sr=sr, frame_length=1024, hop_length=160)
            voiced = f0[np.isfinite(f0)]
            voiced_ratio = float(np.mean(np.isfinite(f0)))
            silent = rms < max(float(np.median(rms))*0.35, 1e-5)
            runs, run = [], 0
            for x in silent:
                if x:
                    run += 1
                elif run:
                    runs.append(run); run = 0
            if run: runs.append(run)
            pauses = [x*160/sr for x in runs if x*160/sr >= .15]
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=400, hop_length=160)[0]
            vm = np.isfinite(f0)
            transitions = int(np.sum(vm[1:] != vm[:-1])) if len(vm) > 1 else 0

            feats = [
                duration, np.mean(rms), np.std(rms),
                np.percentile(rms,95)-np.percentile(rms,5),
                np.mean(zcr),
                np.mean(voiced) if len(voiced) else 0,
                np.std(voiced) if len(voiced) else 0,
                np.median(voiced) if len(voiced) else 0,
                np.percentile(voiced,95)-np.percentile(voiced,5) if len(voiced) else 0,
                voiced_ratio, len(pauses), sum(pauses),
                np.mean(pauses) if pauses else 0,
                transitions/max(duration,1e-6),
                np.mean(centroid), np.std(centroid),
            ]
            # Normalize to fixed dimensionality and retain only finite values.
            feats = np.asarray(feats, dtype=np.float32)
            feats[~np.isfinite(feats)] = 0
            rows_out.append((path, r, feats))
        except Exception as e:
            failures += 1
            print("FAIL", path, e)

        if i % 250 == 0:
            print(f"Prosody {i}/{len(val)}")

    if failures:
        print("Prosody failures:", failures)

    # Load frozen V2 exactly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoiceGuardModel().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    pre = VoiceGuardAudioPreprocessor(target_sr=16000, max_duration_sec=10.0)
    fx = VoiceGuardFeatureExtractor(sample_rate=16000)

    # Produce frozen V2 probabilities on the same validation files.
    data = []
    print("Running frozen V2 validation inference...")
    for i, (path, r, pros) in enumerate(rows_out, 1):
        wav = pre.process(str(path))
        x = fx.extract_with_channel(wav).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits, _ = model(x)
            p_spoof = float(torch.softmax(logits, dim=1)[0,1].item())
        data.append({
            "audio_path": str(path),
            "label": 1 if str(r["label"]).lower()=="spoof" else 0,
            "language": r.get("language",""),
            "condition": r.get("condition",""),
            "generator": r.get("generator",""),
            "v2_spoof_probability": p_spoof,
            "prosody": pros,
        })
        if i % 250 == 0:
            print(f"V2 {i}/{len(rows_out)}")

    y = np.asarray([d["label"] for d in data])
    v2 = np.asarray([d["v2_spoof_probability"] for d in data])
    P = np.asarray([d["prosody"] for d in data])

    # Standardize prosody using validation statistics, then fit a tiny logistic
    # regression using deterministic gradient descent on validation only.
    # This is a diagnostic fusion experiment, NOT a production model.
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    sd[sd < 1e-6] = 1
    Z = (P-mu)/sd
    # Clip extreme values for numerical stability.
    Z = np.clip(Z, -6, 6)

    X = np.column_stack([np.ones(len(Z)), Z])
    w = np.zeros(X.shape[1], dtype=np.float64)
    lr = 0.03
    reg = 1e-2
    for _ in range(1500):
        z = np.clip(X @ w, -30, 30)
        q = 1/(1+np.exp(-z))
        grad = (X.T @ (q-y))/len(y)
        grad[1:] += reg*w[1:]
        w -= lr*grad

    prosody_prob = 1/(1+np.exp(-np.clip(X@w, -30, 30)))

    # Calibrate a simple convex evidence fusion around the frozen V2 probability.
    # We search alpha only; no retraining of V2. alpha=0 means V2 alone.
    # The fused probability is logit(V2)+alpha*logit(prosody).
    def logit(a):
        a = np.clip(a, 1e-5, 1-1e-5)
        return np.log(a/(1-a))
    def sigmoid(a):
        return 1/(1+np.exp(-np.clip(a,-30,30)))

    best = None
    for alpha in np.linspace(-1, 1, 401):
        fused = sigmoid(logit(v2) + alpha*logit(prosody_prob))
        m = metrics(fused, y, 0.25)
        if best is None or m["f1"] > best["metrics"]["f1"]:
            best = {"alpha": float(alpha), "metrics": m}

    fused = sigmoid(logit(v2) + best["alpha"]*logit(prosody_prob))

    # Also report V2-only at threshold 0.25 and 0.5 for context.
    report = {
        "experiment_version": "voiceguard_v2_prosody_incremental_validation_v1",
        "validation_only": True,
        "protected_test_used": False,
        "source_data_unchanged": True,
        "validation_count": len(data),
        "validation_bonafide": int(np.sum(y==0)),
        "validation_spoof": int(np.sum(y==1)),
        "v2_checkpoint": str(CHECKPOINT),
        "v2_checkpoint_epoch": ckpt.get("epoch"),
        "v2_threshold_for_comparison": 0.25,
        "baseline_v2_threshold_025": metrics(v2, y, .25),
        "baseline_v2_threshold_050": metrics(v2, y, .50),
        "prosody_only_threshold_025": metrics(prosody_prob, y, .25),
        "fusion": {
            "method": "logit(v2_probability) + alpha * logit(prosody_probability)",
            "alpha_selected_on_validation": best["alpha"],
            "metrics_threshold_025": best["metrics"],
            "decision": (
                "KEEP_AS_CANDIDATE_EVIDENCE" if best["metrics"]["f1"] > metrics(v2,y,.25)["f1"]
                else "DO_NOT_DEPLOY_PROSODY_FUSION"
            ),
        },
        "warning": (
            "This is a validation-only diagnostic fusion experiment. The prosody "
            "logistic model and fusion alpha are fit on the same validation set, "
            "so these metrics are not an unbiased generalization estimate. "
            "The protected clean test was not used and V2 checkpoint was not modified."
        ),
        "prosody_feature_names": FEATURES,
    }

    # Save per-file predictions.
    with PRED_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = ["audio_path","label","language","condition","generator",
                  "v2_spoof_probability","prosody_probability","fused_probability"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d, pp, ff in zip(data, prosody_prob, fused):
            writer.writerow({
                "audio_path": d["audio_path"],
                "label": d["label"],
                "language": d["language"],
                "condition": d["condition"],
                "generator": d["generator"],
                "v2_spoof_probability": float(d["v2_spoof_probability"]),
                "prosody_probability": float(pp),
                "fused_probability": float(ff),
            })

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("PROSODY INCREMENTAL EXPERIMENT COMPLETE")
    print("=" * 72)
    print("Validation:", len(data))
    print("V2 @0.25 :", report["baseline_v2_threshold_025"])
    print("V2 @0.50 :", report["baseline_v2_threshold_050"])
    print("Prosody  :", report["prosody_only_threshold_025"])
    print("Fusion α  :", best["alpha"])
    print("Fusion    :", best["metrics"])
    print("Decision  :", report["fusion"]["decision"])
    print("CSV       :", PRED_CSV)
    print("Report    :", REPORT_JSON)
    print("=" * 72)


if __name__ == "__main__":
    main()
