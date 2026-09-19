"""
VoiceGuard V2 Full Evidence Integration Test V2.

After the deployable prosody scorer is built:
    V2 -> prosody scorer -> quality -> active risk engine

Uses one train bonafide and one train/validation spoof sample.
Does not use protected TEST.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import librosa

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from inference import VoiceGuardInference
from voiceguard_audio_quality_interface_v1 import assess_audio_quality
from voiceguard_prosody_fusion_v1 import load_scorer, extract_prosody_features
from voiceguard_dynamic_risk_engine_v2 import RiskInput, VoiceGuardRiskEngine

MASTER = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
V2 = ROOT / "experiments" / "reverb_v2" / "models" / "best_model.pt"
SCORER = ROOT / "experiments" / "reverb_v2" / "prosody_fusion_v1" / "prosody_scorer_v1.pkl"

def pick():
    bon = spoof = None
    with MASTER.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "validation"}:
                continue
            p = Path(row["processed_audio_path"])
            if not p.is_absolute():
                p = ROOT / p
            if not p.exists():
                continue
            lab = row["label"].lower()
            if lab == "bonafide" and bon is None:
                bon = (row, p)
            if lab == "spoof" and spoof is None:
                spoof = (row, p)
            if bon and spoof:
                return [bon, spoof]
    raise RuntimeError("Could not find both bonafide and spoof non-test samples.")

def main():
    print("=" * 72)
    print("VOICEGUARD — V2 FULL EVIDENCE INTEGRATION TEST V2")
    print("=" * 72)

    if not SCORER.exists():
        raise FileNotFoundError(
            "Prosody scorer missing. Run build_voiceguard_prosody_fusion_v1.py first."
        )

    v2 = VoiceGuardInference(checkpoint_path=V2)
    scorer = load_scorer(SCORER)
    config = json.loads(SCORER.with_suffix(".json").read_text(encoding="utf-8"))
    engine = VoiceGuardRiskEngine(
        threshold=0.25,
        primary_weight=config["primary_weight"],
        prosody_weight=config["prosody_weight"],
    )

    for row, path in pick():
        wav, sr = librosa.load(str(path), sr=16000, mono=True)
        pred = v2.predict(path)
        quality = assess_audio_quality(wav, sample_rate=sr)
        pf = extract_prosody_features(wav, sr)
        pp = scorer.predict_proba(pf)

        risk = engine.evaluate(
            RiskInput(
                spoof_probability=pred["spoof_probability"],
                quality_confidence_multiplier=quality.confidence_multiplier,
                prosody_probability=pp,
                prosody_reliability=1.0,
                evidence_confidence=1.0,
            )
        )

        print("\nFILE:", row["file_id"], "| LABEL:", row["label"])
        print("V2 spoof probability :", pred["spoof_probability"])
        print("Prosody probability  :", pp)
        print("Quality              :", quality.status)
        print("Quality confidence   :", quality.confidence_multiplier)
        print("Risk score           :", risk.risk_score)
        print("Risk level           :", risk.risk_level)
        print("Status               :", risk.status)

        assert 0 <= pred["spoof_probability"] <= 1
        assert 0 <= pp <= 1
        assert 0 <= risk.risk_score <= 100

    print("\n" + "=" * 72)
    print("Frozen V2 inference       : PASS")
    print("Deployable prosody scorer : PASS")
    print("Audio quality interface   : PASS")
    print("Active evidence fusion    : PASS")
    print("Dynamic risk engine       : PASS")
    print("Protected TEST used       : NO")
    print("Model/threshold tuning    : NO")
    print("=" * 72)
    print("VOICEGUARD V2 FULL EVIDENCE INTEGRATION TEST PASS")
    print("=" * 72)

    out = ROOT / "experiments" / "reverb_v2" / "evidence_integration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "V2_FULL_EVIDENCE_INTEGRATION_TEST_V2.json").write_text(
        json.dumps({
            "status": "PASS",
            "protected_test_used": False,
            "active_components": [
                "frozen_v2",
                "deployable_prosody_scorer",
                "audio_quality",
                "dynamic_risk_engine_v2",
            ],
        }, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
