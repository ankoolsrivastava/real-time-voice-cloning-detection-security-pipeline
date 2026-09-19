from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app import config


# Frozen handoff runtime — read/use only.
if str(config.ML_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(config.ML_RUNTIME_ROOT))

if str(config.ML_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(config.ML_SCRIPTS_ROOT))


from audio_preprocessor import VoiceGuardAudioPreprocessor
from feature_extractor import VoiceGuardFeatureExtractor
from model import VoiceGuardModel

from voiceguard_prosody_fusion_v1 import (ProsodyScorer, extract_prosody_features, load_scorer)

from voiceguard_audio_quality_interface_v1 import (
    AudioQualityResult,
    assess_audio_quality,
)

from voiceguard_dynamic_risk_engine_v2 import (
    RiskInput,
    RiskOutput,
    VoiceGuardRiskEngine,
)


@dataclass(frozen=True)
class MLRuntimeResult:
    prediction: int
    prediction_label: str

    bonafide_probability: float
    spoof_probability: float

    prosody_probability: Optional[float]
    prosody_reliability: float

    quality: AudioQualityResult
    risk: RiskOutput

    device: str
    model_version: str


class VoiceGuardMLRuntime:
    """
    Persistent VoiceGuard V2 ML runtime.

    All frozen ML artifacts are loaded once and reused.
    The backend owns transport, streaming, buffering and sessions.
    The frozen ML handoff owns ML preprocessing, detection,
    prosody, quality, evidence and risk computation.
    """

    MODEL_VERSION = "voiceguard_v2_epoch8"

    def __init__(self) -> None:
        config.validate_paths()

        self._lock = threading.Lock()

        self.device = torch.device(config.DEVICE)

        # ---------------------------------------------------------
        # Frozen V2 preprocessing
        # ---------------------------------------------------------
        self.preprocessor = VoiceGuardAudioPreprocessor(
            target_sr=config.SAMPLE_RATE,
            max_duration_sec=config.WINDOW_DURATION_SEC,
        )

        # ---------------------------------------------------------
        # Frozen V2 feature extraction
        # ---------------------------------------------------------
        self.feature_extractor = VoiceGuardFeatureExtractor(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX,
        )

        # ---------------------------------------------------------
        # Frozen V2 model
        # ---------------------------------------------------------
        self.model = VoiceGuardModel(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
        )

        checkpoint = torch.load(
            config.MODEL_PATH,
            map_location=self.device,
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------
        # Frozen deployable prosody scorer
        # ---------------------------------------------------------
        self.prosody_scorer = load_scorer(config.PROSODY_SCORER_PATH)

        # ---------------------------------------------------------
        # Frozen V2 risk engine
        # ---------------------------------------------------------
        # Fusion weights are frozen from the validated V2
        # PROSODY_FUSION_CONFIG_V1 artifact.
        prosody_config = config.load_prosody_config()

        self.risk_engine = VoiceGuardRiskEngine(
            threshold=config.SPOOF_THRESHOLD,
            primary_weight=float(prosody_config["primary_weight"]),
            prosody_weight=float(prosody_config["prosody_weight"]),
        )

    @torch.inference_mode()
    def _predict_v2(
        self,
        waveform: np.ndarray,
    ) -> tuple[float, float, int]:

        # Match the frozen V2 inference path exactly:
        # extract_with_channel() returns [1, 64, time].
        features = self.feature_extractor.extract_with_channel(waveform)

        if isinstance(features, np.ndarray):
            tensor = torch.from_numpy(features).float()
        else:
            tensor = torch.as_tensor(features).float()

        # Add the batch dimension:
        # [1, 64, time] -> [1, 1, 64, time]
        if tensor.ndim != 3:
            raise ValueError(
                "Expected V2 feature tensor with shape "
                f"[1, 64, time], got {tuple(tensor.shape)}"
            )

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        output = self.model(tensor)

        if isinstance(output, tuple):
            logits = output[0]
        elif isinstance(output, dict):
            logits = output.get("logits", output["output"])
        else:
            logits = output

        probabilities = torch.softmax(logits, dim=-1)[0]

        bonafide_probability = float(
            probabilities[0].item()
        )

        spoof_probability = float(
            probabilities[1].item()
        )

        prediction = int(
            torch.argmax(probabilities).item()
        )

        return (
            bonafide_probability,
            spoof_probability,
            prediction,
        )

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        packet_loss_ratio: Optional[float] = None,
        jitter_ms: Optional[float] = None,
        codec_degradation_score: Optional[float] = None,
    ) -> MLRuntimeResult:
        """
        Run the complete frozen VoiceGuard evidence pipeline
        on an in-memory waveform.

        The backend is responsible for constructing the exact
        V2 window before calling this method.
        """

        with self._lock:

            waveform = np.asarray(
                waveform,
                dtype=np.float32,
            )

            if waveform.ndim > 1:
                waveform = np.mean(
                    waveform,
                    axis=0,
                )

            if sample_rate != config.SAMPLE_RATE:
                raise ValueError(
                    f"Expected {config.SAMPLE_RATE} Hz audio; "
                    f"received {sample_rate} Hz."
                )

            # Exact V2 preprocessing operations.
            waveform = self.preprocessor.normalize(
                waveform
            )

            waveform = self.preprocessor.pad_or_truncate(
                waveform
            )

            # Frozen V2 detector.
            (
                bonafide_probability,
                spoof_probability,
                prediction,
            ) = self._predict_v2(waveform)

            # Frozen audio quality interface.
            quality = assess_audio_quality(
                waveform,
                sample_rate=config.SAMPLE_RATE,
                packet_loss_ratio=packet_loss_ratio,
                jitter_ms=jitter_ms,
                codec_degradation_score=codec_degradation_score,
            )

            # Frozen deployable prosody.
            prosody_features = extract_prosody_features(
                waveform,
                sr=config.SAMPLE_RATE,
            )

            prosody_probability = float(
                self.prosody_scorer.predict_proba(
                    prosody_features
                )
            )

            prosody_reliability = 1.0

            # Frozen V2 risk engine.
            risk = self.risk_engine.evaluate(
                RiskInput(
                    spoof_probability=spoof_probability,
                    quality_confidence_multiplier=(
                        quality.confidence_multiplier
                    ),
                    prosody_probability=prosody_probability,
                    prosody_reliability=prosody_reliability,
                    evidence_confidence=(
                        quality.confidence_multiplier
                    ),
                )
            )

            return MLRuntimeResult(
                prediction=prediction,
                prediction_label=(
                    "spoof"
                    if prediction == 1
                    else "bonafide"
                ),
                bonafide_probability=bonafide_probability,
                spoof_probability=spoof_probability,
                prosody_probability=prosody_probability,
                prosody_reliability=prosody_reliability,
                quality=quality,
                risk=risk,
                device=str(self.device),
                model_version=self.MODEL_VERSION,
            )


_runtime: Optional[VoiceGuardMLRuntime] = None
_runtime_lock = threading.Lock()


def get_ml_runtime() -> VoiceGuardMLRuntime:
    """
    Return the process-wide persistent ML runtime.

    The frozen model and supporting components are loaded only
    once per backend process.
    """

    global _runtime

    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = VoiceGuardMLRuntime()

    return _runtime




