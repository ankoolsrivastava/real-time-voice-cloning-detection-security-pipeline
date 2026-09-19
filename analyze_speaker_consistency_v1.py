from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(r"D:\VoiceGaurd")
MANIFEST = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
OUT = ROOT / "experiments" / "reverb_v2" / "speaker_consistency"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS_CSV = OUT / "SPEAKER_CONSISTENCY_PAIRS.csv"
REPORT_JSON = OUT / "SPEAKER_CONSISTENCY_REPORT.json"

SEED = 42
SR = 16000


def resolve_audio(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    p = ROOT / raw
    if p.exists():
        return p
    normalized = raw.replace("/", "\\")
    marker = r"D:\VoiceGaurd"
    if normalized.lower().startswith(marker.lower()):
        p = ROOT / normalized[len(marker):].lstrip("\\")
        if p.exists():
            return p
    return None


def embedding(path: Path) -> np.ndarray:
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError("empty audio")

    # Use the same fixed 10 s convention as VoiceGuard for consistency.
    target = SR * 10
    if len(y) > target:
        y = y[:target]
    elif len(y) < target:
        y = np.pad(y, (0, target - len(y)))

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, n_fft=400, hop_length=160, win_length=400
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Statistics pooling creates a fixed speaker representation.
    emb = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ]).astype(np.float32)

    norm = np.linalg.norm(emb)
    if norm == 0:
        raise ValueError("zero embedding")
    return emb / norm


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def roc_auc(scores, labels):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None
    tpr = np.cumsum(y == 1) / pos
    fpr = np.cumsum(y == 0) / neg
    return float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))


def eer(scores, labels):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    fn = pos - tp
    tn = neg - fp
    fpr = fp / neg
    fnr = fn / pos
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    print("=" * 72)
    print("VOICEGUARD — CROSS-SESSION SPEAKER CONSISTENCY")
    print("=" * 72)

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"speaker_id", "split", "label", "language", "processed_audio_path"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise RuntimeError(f"Missing required manifest fields: {sorted(missing)}")

    # Validation-only genuine speech. We intentionally do not use the protected
    # clean test split for this diagnostic.
    candidates = []
    for r in rows:
        if str(r.get("label", "")).lower() != "bonafide":
            continue
        if str(r.get("split", "")).lower() != "validation":
            continue
        speaker = str(r.get("speaker_id", "")).strip()
        p = resolve_audio(str(r.get("processed_audio_path", "")).strip())
        if speaker and speaker.lower() != "nan" and p is not None:
            candidates.append((speaker, p, r))

    by_speaker = {}
    for item in candidates:
        by_speaker.setdefault(item[0], []).append(item)

    # Need at least two genuine clips per speaker for an enrollment/query pair.
    eligible = {s: v for s, v in by_speaker.items() if len(v) >= 2}
    if len(eligible) < 10:
        raise RuntimeError(f"Only {len(eligible)} speakers have >=2 validation clips.")

    rng = random.Random(SEED)
    speakers = sorted(eligible)

    # Cache one embedding per audio file.
    cache = {}
    failures = 0

    def get_emb(path):
        nonlocal failures
        key = str(path)
        if key not in cache:
            try:
                cache[key] = embedding(path)
            except Exception as e:
                failures += 1
                print("Embedding failure:", path, e)
                cache[key] = None
        return cache[key]

    records = []

    # Positive: same speaker, separate clips.
    # Negative: enrollment from speaker A, query from speaker B.
    # Use deterministic 1:1 sampling to avoid class imbalance.
    positive_pairs = []
    for speaker in speakers:
        items = eligible[speaker]
        rng.shuffle(items)
        for i in range(min(3, len(items) - 1)):
            a = items[i]
            b = items[i + 1]
            positive_pairs.append((speaker, a, b))

    rng.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:500]

    for speaker, a, b in positive_pairs:
        ea = get_emb(a[1])
        eb = get_emb(b[1])
        if ea is None or eb is None:
            continue
        records.append({
            "pair_type": "same_speaker",
            "enrollment_speaker": speaker,
            "query_speaker": speaker,
            "enrollment_path": str(a[1]),
            "query_path": str(b[1]),
            "language_enrollment": a[2].get("language", ""),
            "language_query": b[2].get("language", ""),
            "condition_enrollment": a[2].get("condition", ""),
            "condition_query": b[2].get("condition", ""),
            "similarity": cosine(ea, eb),
            "target": 1,
        })

    # Negative pairs, matched approximately by language when possible.
    for _ in range(len(records)):
        s1 = rng.choice(speakers)
        possible = [s for s in speakers if s != s1]
        s2 = rng.choice(possible)
        a = rng.choice(eligible[s1])
        b_candidates = eligible[s2]

        # Prefer same language negative pairs so language isn't the primary cue.
        same_lang = [x for x in b_candidates if x[2].get("language", "") == a[2].get("language", "")]
        b = rng.choice(same_lang or b_candidates)

        ea = get_emb(a[1])
        eb = get_emb(b[1])
        if ea is None or eb is None:
            continue
        records.append({
            "pair_type": "different_speaker",
            "enrollment_speaker": s1,
            "query_speaker": s2,
            "enrollment_path": str(a[1]),
            "query_path": str(b[1]),
            "language_enrollment": a[2].get("language", ""),
            "language_query": b[2].get("language", ""),
            "condition_enrollment": a[2].get("condition", ""),
            "condition_query": b[2].get("condition", ""),
            "similarity": cosine(ea, eb),
            "target": 0,
        })

    if len(records) < 100:
        raise RuntimeError(f"Too few valid pairs: {len(records)}")

    with PAIRS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    same = np.asarray([r["similarity"] for r in records if r["target"] == 1], dtype=float)
    diff = np.asarray([r["similarity"] for r in records if r["target"] == 0], dtype=float)
    scores = [r["similarity"] for r in records]
    labels = [r["target"] for r in records]

    # Threshold sweep for a descriptive operating point.
    best = None
    for threshold in np.linspace(-1, 1, 401):
        pred = np.asarray([1 if s >= threshold else 0 for s in scores])
        y = np.asarray(labels)
        tp = int(np.sum((pred == 1) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        acc = (tp + tn) / len(y)
        if best is None or acc > best["accuracy"]:
            best = {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }

    report = {
        "analysis_version": "voiceguard_cross_session_speaker_consistency_v1",
        "manifest": str(MANIFEST),
        "split_used": "validation",
        "label_used": "bonafide_only",
        "protected_test_used": False,
        "source_data_unchanged": True,
        "representation": {
            "type": "MFCC_statistics_speaker_representation",
            "sample_rate": SR,
            "n_mfcc": 40,
            "n_fft": 400,
            "hop_length": 160,
            "statistics": "mean_and_std_of_MFCC_delta_delta2",
            "normalization": "L2",
            "similarity": "cosine",
        },
        "eligible_speakers": len(eligible),
        "audio_embeddings_computed": len([x for x in cache.values() if x is not None]),
        "embedding_failures": failures,
        "same_speaker_pairs": int(len(same)),
        "different_speaker_pairs": int(len(diff)),
        "same_speaker_similarity": {
            "mean": float(np.mean(same)),
            "median": float(np.median(same)),
            "std": float(np.std(same)),
            "p05": float(np.percentile(same, 5)),
            "p95": float(np.percentile(same, 95)),
        },
        "different_speaker_similarity": {
            "mean": float(np.mean(diff)),
            "median": float(np.median(diff)),
            "std": float(np.std(diff)),
            "p05": float(np.percentile(diff, 5)),
            "p95": float(np.percentile(diff, 95)),
        },
        "separation": {
            "mean_gap_same_minus_different": float(np.mean(same) - np.mean(diff)),
            "roc_auc": roc_auc(scores, labels),
            "eer": eer(scores, labels),
            "best_descriptive_accuracy_threshold": best,
        },
        "architecture_note": (
            "This V1 module is a lightweight speaker representation diagnostic, "
            "not a pretrained production-grade speaker verification model. "
            "It demonstrates the cross-session consistency evidence interface "
            "without downloading or modifying a large external model."
        ),
        "no_reference_behavior": (
            "When no trusted historical reference exists, speaker consistency "
            "must be returned as unavailable rather than inferred."
        ),
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("SPEAKER CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 72)
    print("Eligible speakers :", len(eligible))
    print("Same-speaker pairs:", len(same))
    print("Different pairs   :", len(diff))
    print("Embedding failures:", failures)
    print(f"Same mean         : {np.mean(same):.4f}")
    print(f"Different mean    : {np.mean(diff):.4f}")
    print(f"Mean similarity gap: {np.mean(same)-np.mean(diff):.4f}")
    print(f"ROC-AUC           : {roc_auc(scores, labels):.4f}")
    print(f"EER               : {eer(scores, labels):.4f}")
    print(f"Best threshold    : {best['threshold']:.4f}")
    print(f"Best accuracy     : {best['accuracy']:.4f}")
    print("CSV               :", PAIRS_CSV)
    print("Report            :", REPORT_JSON)
    print("=" * 72)


if __name__ == "__main__":
    main()
