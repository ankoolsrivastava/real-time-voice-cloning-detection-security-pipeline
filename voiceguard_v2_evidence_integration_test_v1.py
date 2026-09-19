"""
VoiceGuard V2 Evidence Integration Test V1

Runs a REAL integration smoke test against the frozen V2 checkpoint
using non-test samples from the locked master manifest.

Pipeline:
    real WAV
      -> frozen V2 inference
      -> audio quality
      -> prosody diagnostics
      -> dynamic risk engine

Important:
- Does NOT modify any dataset.
- Does NOT use the protected test split.
- Does NOT tune thresholds.
- Prosody is exposed as diagnostic evidence metadata only here because
  the validation-trained prosody scorer was not packaged as a deployable
  calibrated model. The risk engine therefore correctly treats prosody
  as unavailable in this integration test rather than inventing a score.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# Make the root and scripts importable without changing project files.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from voiceguard_dynamic_risk_engine_v1 import RiskInput, VoiceGuardRiskEngine
from voiceguard_audio_quality_interface_v1 import assess_audio_quality
from inference import VoiceGuardInference


MASTER = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
CHECKPOINT = ROOT / "experiments" / "reverb_v2" / "models" / "best_model.pt"


def choose_non_test_rows(limit: int = 2):
    if not MASTER.exists():
        raise FileNotFoundError(f"Master manifest not found: {MASTER}")

    rows = []
    with MASTER.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # The locked master manifest does not require a column named
        # `audio_path`. Resolve the actual audio-file column from the header
        # without modifying the manifest.
        fieldnames = reader.fieldnames or []
        audio_column = None

        preferred = [
            "audio_path", "processed_path", "file_path", "path",
            "wav_path", "audio_file", "filepath", "file"
        ]
        for candidate in preferred:
            if candidate in fieldnames:
                audio_column = candidate
                break

        if audio_column is None:
            # Last-resort schema-safe discovery: find a column whose values
            # look like WAV paths and point to existing files.
            sample_rows = []
            for row in reader:
                sample_rows.append(row)
                if len(sample_rows) >= 25:
                    break

            for candidate in fieldnames:
                for sample in sample_rows:
                    value = (sample.get(candidate) or "").strip()
                    if value.lower().endswith(".wav"):
                        candidate_path = Path(value)
                        if not candidate_path.is_absolute():
                            candidate_path = ROOT / candidate_path
                        if candidate_path.exists():
                            audio_column = candidate
                            break
                if audio_column:
                    break

            # Re-open below because the initial reader was consumed.
            f.close()
            f = MASTER.open("r", encoding="utf-8-sig", newline="")
            reader = csv.DictReader(f)

        if audio_column is None:
            raise RuntimeError(
                "Could not resolve an audio-file column in the locked master "
                f"manifest. Columns found: {fieldnames}"
            )

        for row in reader:
            if row.get("split") in {"train", "validation"}:
                value = (row.get(audio_column) or "").strip()
                path = Path(value)
                if not path.is_absolute():
                    path = ROOT / path
                if path.exists():
                    rows.append((row, path))
                    if len(rows) >= limit:
                        break

    if len(rows) < limit:
        raise RuntimeError("Could not find enough existing non-test audio files.")

    return rows


def prosody_diagnostics(waveform: np.ndarray, sr: int = 16000):
    """Descriptive prosody features only; no classification score is invented."""
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

    active = rms > max(np.percentile(rms, 35) * 0.8, 1e-5)
    voiced_ratio = float(np.mean(active))

    transitions = np.sum(active[1:] != active[:-1]) if len(active) > 1 else 0
    transition_rate = float(transitions / max(len(active) - 1, 1))

    return {
        "duration_sec": float(len(y) / sr),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "voiced_ratio": voiced_ratio,
        "voiced_transition_rate": transition_rate,
        "zcr_mean": float(np.mean(zcr)),
    }


def main():
    print("=" * 72)
    print("VOICEGUARD — V2 EVIDENCE INTEGRATION TEST V1")
    print("=" * 72)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Frozen V2 checkpoint not found: {CHECKPOINT}")

    rows = choose_non_test_rows(limit=2)
    engine = VoiceGuardInference(checkpoint_path=CHECKPOINT)
    risk_engine = VoiceGuardRiskEngine(threshold=0.25)

    results = []

    for i, (row, audio_path) in enumerate(rows, start=1):
        print(f"\n--- SAMPLE {i} ---")
        print("File ID :", row.get("file_id"))
        print("Split   :", row.get("split"))
        print("Label   :", row.get("label"))
        print("Audio   :", audio_path)

        prediction = engine.predict(audio_path)

        waveform, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        quality = assess_audio_quality(waveform, sample_rate=sr)
        prosody = prosody_diagnostics(waveform, sr=sr)

        # Deliberately do not manufacture a prosody classifier score.
        risk = risk_engine.evaluate(
            RiskInput(
                spoof_probability=prediction["spoof_probability"],
                quality_confidence_multiplier=quality.confidence_multiplier,
                prosody_evidence=None,
                prosody_reliability=0.0,
                evidence_confidence=1.0,
            )
        )

        print("V2 spoof probability :", prediction["spoof_probability"])
        print("V2 prediction        :", prediction["prediction"])
        print("Quality status       :", quality.status)
        print("Quality confidence   :", quality.confidence_multiplier)
        print("Prosody diagnostics  :", json.dumps(prosody, indent=2))
        print("Prosody fusion       : NOT USED (diagnostic-only)")
        print("Risk score           :", risk.risk_score)
        print("Risk level           :", risk.risk_level)
        print("Risk status          :", risk.status)

        assert prediction["feature_shape"] == [1, 1, 64, 1001]
        assert 0.0 <= prediction["spoof_probability"] <= 1.0
        assert 0.0 <= quality.confidence_multiplier <= 1.0
        assert 0.0 <= risk.risk_score <= 100.0

        results.append({
            "file_id": row.get("file_id"),
            "split": row.get("split"),
            "label": row.get("label"),
            "spoof_probability": prediction["spoof_probability"],
            "quality_status": quality.status,
            "quality_confidence_multiplier": quality.confidence_multiplier,
            "prosody_status": "diagnostic_only",
            "risk_score": risk.risk_score,
            "risk_level": risk.risk_level,
        })

    print("\n" + "=" * 72)
    print("INTEGRATION TEST RESULT")
    print("=" * 72)
    print("Frozen V2 inference              : PASS")
    print("Audio quality interface          : PASS")
    print("Prosody diagnostics interface    : PASS")
    print("Risk engine integration          : PASS")
    print("Protected test split used        : NO")
    print("Threshold/model tuning performed : NO")
    print("Synthetic/dummy audio used       : NO")
    print("PROSODY CLASSIFIER FUSION         : NOT CLAIMED")
    print("=" * 72)
    print("VOICEGUARD V2 EVIDENCE INTEGRATION TEST PASS")
    print("=" * 72)

    report_dir = ROOT / "experiments" / "reverb_v2" / "evidence_integration"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "V2_EVIDENCE_INTEGRATION_TEST_V1.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "protected_test_used": False,
                "threshold_tuning": False,
                "synthetic_audio": False,
                "frozen_checkpoint": str(CHECKPOINT),
                "prosody_fusion": "not_claimed_diagnostic_only",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Report:", report)


if __name__ == "__main__":
    main()
