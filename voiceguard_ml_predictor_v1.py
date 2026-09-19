"""
VoiceGuard ML Production Predictor V1

Public ML-to-system inference interface.

Pipeline:

    PCM float32 audio window
        ↓
    VoiceGuard preprocessing
        ↓
    Frozen VoiceGuard V2
        ↓
    Prosody scorer
        ↓
    Audio quality
        ↓
    Evidence fusion / Dynamic Risk Engine
        ↓
    Contract-compliant result

Important:
- Does not train anything.
- Does not modify the frozen V2 model.
- Does not use the protected TEST split.
- Does not implement microphone/call streaming.
- Speaker consistency remains UNAVAILABLE unless a trusted
  reference implementation is explicitly added later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ------------------------------------------------------------------
# Project paths / imports
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from inference import VoiceGuardInference
from audio_preprocessor import VoiceGuardAudioPreprocessor
from voiceguard_audio_quality_interface_v1 import assess_audio_quality
from voiceguard_prosody_fusion_v1 import (
    extract_prosody_features,
    load_scorer,
)
from voiceguard_dynamic_risk_engine_v2 import (
    RiskInput,
    VoiceGuardRiskEngine,
)


# ------------------------------------------------------------------
# Frozen production artifacts
# ------------------------------------------------------------------

MODEL_VERSION = "voiceguard_v2_epoch8"

CHECKPOINT_PATH = (
    ROOT
    / "experiments"
    / "reverb_v2"
    / "models"
    / "best_model.pt"
)

PROSODY_SCORER_PATH = (
    ROOT
    / "experiments"
    / "reverb_v2"
    / "prosody_fusion_v1"
    / "prosody_scorer_v1.pkl"
)

PROSODY_METADATA_PATH = (
    PROSODY_SCORER_PATH.with_suffix(".json")
)

CONTRACT_PATH = (
    ROOT
    / "voiceguard_ml_evidence_contract_v1.json"
)

SAMPLE_RATE = 16000
CHANNELS = 1
MAX_WINDOW_DURATION_SEC = 10.0
MAX_SAMPLES = int(
    SAMPLE_RATE * MAX_WINDOW_DURATION_SEC
)

THRESHOLD = 0.25


class VoiceGuardMLPredictor:
    """
    Production-facing VoiceGuard ML predictor.

    The backend/system layer supplies one audio window.
    This class performs the ML/evidence/risk processing and
    returns one structured result.
    """

    def __init__(
        self,
        device: Optional[str] = None,
    ):
        self.model_version = MODEL_VERSION
        self.threshold = THRESHOLD

        # ----------------------------------------------------------
        # Validate required production artifacts
        # ----------------------------------------------------------

        self._require_file(
            CHECKPOINT_PATH,
            "Frozen V2 checkpoint",
        )

        self._require_file(
            PROSODY_SCORER_PATH,
            "Prosody scorer artifact",
        )

        self._require_file(
            PROSODY_METADATA_PATH,
            "Prosody scorer metadata",
        )

        # Contract is documentation/configuration, but its presence
        # is useful for a production installation sanity check.
        self._require_file(
            CONTRACT_PATH,
            "ML evidence contract",
        )

        # ----------------------------------------------------------
        # Load frozen V2 detector
        # ----------------------------------------------------------

        self.v2 = VoiceGuardInference(
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
        )

        # ----------------------------------------------------------
        # Load frozen prosody scorer
        # ----------------------------------------------------------

        self.prosody_scorer = load_scorer(
            PROSODY_SCORER_PATH
        )

        self.prosody_metadata = json.loads(
            PROSODY_METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        # ----------------------------------------------------------
        # Read validated fusion configuration from the frozen
        # prosody artifact metadata.
        # ----------------------------------------------------------

        self.primary_weight = float(
            self.prosody_metadata["primary_weight"]
        )

        self.prosody_weight = float(
            self.prosody_metadata["prosody_weight"]
        )

        # The production predictor must not silently use the
        # risk engine's unrelated defaults.
        if not np.isclose(
            self.primary_weight + self.prosody_weight,
            1.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Invalid frozen fusion weights: "
                "weights must sum to 1.0."
            )

        # ----------------------------------------------------------
        # Active validated risk configuration
        # ----------------------------------------------------------

        self.risk_engine = VoiceGuardRiskEngine(
            threshold=self.threshold,
            primary_weight=self.primary_weight,
            prosody_weight=self.prosody_weight,
        )

        # ----------------------------------------------------------
        # Same preprocessing contract used by VoiceGuard
        # ----------------------------------------------------------

        self.preprocessor = (
            VoiceGuardAudioPreprocessor(
                target_sr=SAMPLE_RATE,
                max_duration_sec=MAX_WINDOW_DURATION_SEC,
            )
        )

    # ==============================================================
    # Helpers
    # ==============================================================

    @staticmethod
    def _require_file(
        path: Path,
        description: str,
    ) -> None:

        if not path.exists():
            raise FileNotFoundError(
                f"{description} does not exist:\n{path}"
            )

        if not path.is_file():
            raise ValueError(
                f"{description} is not a file:\n{path}"
            )

    @staticmethod
    def _validate_context(
        window_id: str,
        timestamp_start_ms: int,
        timestamp_end_ms: int,
        sequence_id: int,
    ) -> None:

        if not isinstance(window_id, str):
            raise TypeError(
                "window_id must be a string."
            )

        if not window_id.strip():
            raise ValueError(
                "window_id must not be empty."
            )

        if not isinstance(
            timestamp_start_ms,
            (int, np.integer),
        ):
            raise TypeError(
                "timestamp_start_ms must be an integer."
            )

        if not isinstance(
            timestamp_end_ms,
            (int, np.integer),
        ):
            raise TypeError(
                "timestamp_end_ms must be an integer."
            )

        if timestamp_start_ms < 0:
            raise ValueError(
                "timestamp_start_ms must be >= 0."
            )

        if timestamp_end_ms < timestamp_start_ms:
            raise ValueError(
                "timestamp_end_ms must be >= "
                "timestamp_start_ms."
            )

        if not isinstance(
            sequence_id,
            (int, np.integer),
        ):
            raise TypeError(
                "sequence_id must be an integer."
            )

        if sequence_id < 0:
            raise ValueError(
                "sequence_id must be >= 0."
            )

    @staticmethod
    def _validate_audio(
        waveform,
    ) -> np.ndarray:

        y = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if y.ndim != 1:
            raise ValueError(
                "Audio input must be mono 1-D PCM."
            )

        if y.size == 0:
            raise ValueError(
                "Audio input is empty."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "Audio input contains NaN or Inf."
            )

        if y.size > MAX_SAMPLES:
            raise ValueError(
                "Audio window exceeds the maximum "
                f"duration of {MAX_WINDOW_DURATION_SEC} seconds."
            )

        return y

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def predict(
        self,
        waveform,
        sample_rate: int = SAMPLE_RATE,
        *,
        window_id: str,
        timestamp_start_ms: int,
        timestamp_end_ms: int,
        sequence_id: int,
        quality_context: Optional[dict[str, Any]] = None,
        speaker_reference: Any = None,
    ) -> dict[str, Any]:
        """
        Run production VoiceGuard inference on one audio window.

        Parameters
        ----------
        waveform:
            Mono PCM waveform. Expected float32-compatible values.

        sample_rate:
            Must be 16000 Hz.

        window_id:
            Backend/system window identifier.

        timestamp_start_ms:
            Window start timestamp.

        timestamp_end_ms:
            Window end timestamp.

        sequence_id:
            Monotonic stream/window sequence identifier.

        quality_context:
            Reserved for externally supplied stream-quality metadata.
            Current baseline quality assessment is performed locally.

        speaker_reference:
            Reserved for future trusted speaker-reference support.
            Currently ignored and reported as unavailable.

        Returns
        -------
        dict
            VoiceGuard ML evidence/risk result.
        """

        # ----------------------------------------------------------
        # Validate input
        # ----------------------------------------------------------

        self._validate_context(
            window_id,
            timestamp_start_ms,
            timestamp_end_ms,
            sequence_id,
        )

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"VoiceGuard production input requires "
                f"{SAMPLE_RATE} Hz audio; received "
                f"{sample_rate} Hz."
            )

        y = self._validate_audio(
            waveform
        )

        notes: list[str] = []

        # ----------------------------------------------------------
        # Preprocess
        # ----------------------------------------------------------

        processed_waveform = (
            self.preprocessor.process_waveform(
                y,
                sample_rate,
            )
        )

        processed_np = (
            processed_waveform
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ----------------------------------------------------------
        # Audio quality
        #
        # Quality is evidence confidence, NOT spoof evidence.
        # ----------------------------------------------------------

        quality_kwargs = {}

        if quality_context is not None:
            if not isinstance(quality_context, dict):
                raise TypeError(
                    "quality_context must be a dictionary or None."
                )

            allowed_quality_fields = {
                "packet_loss_ratio",
                "jitter_ms",
                "codec_degradation_score",
            }

            unknown_fields = (
                set(quality_context.keys())
                - allowed_quality_fields
            )

            if unknown_fields:
                raise ValueError(
                    "Unsupported quality_context fields: "
                    f"{sorted(unknown_fields)}"
                )

            quality_kwargs = {
                key: quality_context[key]
                for key in allowed_quality_fields
                if key in quality_context
            }

        quality = assess_audio_quality(
            processed_np,
            sample_rate=SAMPLE_RATE,
            **quality_kwargs,
        )

        quality_notes = (
            quality.notes or []
        )

        notes.extend(
            str(note)
            for note in quality_notes
        )

        if quality.status == "INVALID":
            notes.append(
                "Audio quality is INVALID; "
                "no strong evidence should be accumulated."
            )

        # ----------------------------------------------------------
        # Frozen V2 inference
        #
        # The existing V2 engine accepts a file path, while the
        # production contract supplies PCM. Therefore we reproduce
        # the already-validated preprocessing/feature path here
        # and invoke the model directly.
        # ----------------------------------------------------------

        features = (
            self.v2.feature_extractor
            .extract_with_channel(
                processed_waveform
            )
        )

        if not np.isfinite(
            features.cpu().numpy()
        ).all():
            raise ValueError(
                "Non-finite Log-Mel features detected."
            )

        features = (
            features
            .unsqueeze(0)
            .to(self.v2.device)
        )

        import torch

        with torch.no_grad():
            logits, attention = (
                self.v2.model(features)
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        bonafide_probability = float(
            probabilities[0, 0]
            .cpu()
            .item()
        )

        spoof_probability = float(
            probabilities[0, 1]
            .cpu()
            .item()
        )

        primary_decision = (
            "SPOOF"
            if spoof_probability >= self.threshold
            else "BONAFIDE"
        )

        # ----------------------------------------------------------
        # Prosody
        # ----------------------------------------------------------

        prosody_features = (
            extract_prosody_features(
                processed_np,
                sr=SAMPLE_RATE,
            )
        )

        try:
            prosody_probability = (
                self.prosody_scorer.predict_proba(
                    prosody_features
                )
            )

            prosody_probability = float(
                np.clip(
                    prosody_probability,
                    0.0,
                    1.0,
                )
            )

            prosody_available = True
            prosody_status = "AVAILABLE"
            prosody_reliability = 1.0

        except Exception as exc:
            prosody_probability = None
            prosody_available = False
            prosody_status = "ERROR"
            prosody_reliability = 0.0

            notes.append(
                "Prosody scorer error: "
                f"{type(exc).__name__}"
            )

        # ----------------------------------------------------------
        # Dynamic risk engine
        # ----------------------------------------------------------

        risk = self.risk_engine.evaluate(
            RiskInput(
                spoof_probability=spoof_probability,
                quality_confidence_multiplier=(
                    quality.confidence_multiplier
                ),
                prosody_probability=(
                    prosody_probability
                ),
                prosody_reliability=(
                    prosody_reliability
                ),
                evidence_confidence=1.0,
            )
        )

        # ----------------------------------------------------------
        # Evidence confidence
        # ----------------------------------------------------------

        evidence_confidence = float(
            np.clip(
                risk.evidence_confidence,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------------
        # Speaker consistency
        #
        # No trusted reference implementation exists in the active
        # pipeline, therefore we explicitly report unavailable.
        # ----------------------------------------------------------

        speaker_consistency = {
            "available": False,
            "similarity": None,
            "evidence": None,
            "status": "UNAVAILABLE",
        }

        notes.append(
            "Speaker consistency unavailable: "
            "no trusted reference is used."
        )

        # ----------------------------------------------------------
        # Optional stream quality context
        # ----------------------------------------------------------

        if quality_context is not None:
            notes.append(
                "External stream-quality context was applied "
                "to the audio quality assessment."
            )

        # ----------------------------------------------------------
        # Final result
        # ----------------------------------------------------------

        result = {
            "model_version": self.model_version,

            "window_id": window_id,

            "sequence_id": int(
                sequence_id
            ),

            "timestamp_start_ms": int(
                timestamp_start_ms
            ),

            "timestamp_end_ms": int(
                timestamp_end_ms
            ),

            "spoof_probability": round(
                spoof_probability,
                6,
            ),

            "threshold": self.threshold,

            "primary_decision": primary_decision,

            "prosody": {
                "available": prosody_available,
                "evidence": (
                    round(
                        prosody_probability,
                        6,
                    )
                    if prosody_probability is not None
                    else None
                ),
                "reliability": (
                    prosody_reliability
                    if prosody_available
                    else None
                ),
                "status": prosody_status,
            },

            "speaker_consistency": (
                speaker_consistency
            ),

            "audio_quality": {
                "quality_score": round(
                    float(
                        quality.quality_score
                    ),
                    6,
                ),
                "confidence_multiplier": round(
                    float(
                        quality.confidence_multiplier
                    ),
                    6,
                ),
                "status": quality.status,
                "rms_db": round(
                    float(quality.rms_db),
                    4,
                ),
                "peak": round(
                    float(quality.peak),
                    6,
                ),
                "clipping_ratio": round(
                    float(quality.clipping_ratio),
                    6,
                ),
                "duration_sec": round(
                    float(quality.duration_sec),
                    6,
                ),
                "active_ratio": round(
                    float(quality.active_ratio),
                    6,
                ),
            },

            "evidence_confidence": (
                round(
                    evidence_confidence,
                    6,
                )
            ),

            "risk": {
                "risk_score": risk.risk_score,
                "risk_level": risk.risk_level,
                "adjusted_spoof_probability": (
                    risk.adjusted_spoof_probability
                ),
                "primary_contribution": (
                    risk.primary_contribution
                ),
                "prosody_contribution": (
                    risk.prosody_contribution
                ),
                "status": risk.status,
                "reasons": list(
                    risk.reasons
                ),
            },

            "notes": notes,
        }

        return result


# ==================================================================
# Safe status / smoke test
# ==================================================================

def main() -> None:

    print("=" * 72)
    print("VOICEGUARD ML PRODUCTION PREDICTOR V1")
    print("=" * 72)

    print()
    print("Model version:", MODEL_VERSION)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Prosody scorer:", PROSODY_SCORER_PATH)
    print("Threshold:", THRESHOLD)

    print()
    print(
        "This module is the ML production interface."
    )

    print(
        "Streaming/microphone capture remains "
        "outside this layer."
    )

    print(
        "Speaker consistency:",
        "UNAVAILABLE without trusted reference.",
    )

    print()
    print("Artifacts verified.")
    print("=" * 72)


if __name__ == "__main__":
    main()