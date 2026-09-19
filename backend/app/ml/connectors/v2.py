from __future__ import annotations

import time

from app.ml.connectors.base import MLModelConnector, MLModelOutput
from app.ml.runtime import MLRuntimeResult, get_ml_runtime


class V2ModelConnector(MLModelConnector):

    @property
    def model_id(self) -> str:
        return "voiceguard_v2"

    @property
    def model_version(self) -> str:
        return "voiceguard_v2_epoch8"

    def predict_waveform(
        self,
        waveform,
        sample_rate: int,
    ) -> MLModelOutput:

        start = time.perf_counter()

        result = get_ml_runtime().predict_waveform(
            waveform,
            sample_rate=sample_rate,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return self.from_runtime_result(result, latency_ms)

    def from_runtime_result(
        self,
        result: MLRuntimeResult,
        latency_ms: float,
    ) -> MLModelOutput:

        return MLModelOutput(
            model_id=self.model_id,
            model_version=self.model_version,
            spoof_probability=float(result.spoof_probability),
            bonafide_probability=float(result.bonafide_probability),
            confidence=float(result.risk.evidence_confidence),
            latency_ms=float(latency_ms),
            status="ok",
            evidence={
                "acoustic_spectral": {
                    "available": True,
                    "spoof_probability": float(result.spoof_probability),
                },
                "prosody_behavioral": {
                    "available": True,
                    "probability": (
                        float(result.risk.prosody_contribution)
                        if result.risk.prosody_contribution is not None
                        else None
                    ),
                    "reliability": float(result.risk.evidence_confidence),
                },
                "speaker_consistency": {
                    "available": False,
                    "reason": "inactive_in_frozen_v2",
                },
            },
            quality={
                "quality_score": float(result.quality.quality_score),
                "confidence_multiplier": float(
                    result.quality.confidence_multiplier
                ),
                "status": result.quality.status,
                "packet_loss_ratio": result.quality.packet_loss_ratio,
                "jitter_ms": result.quality.jitter_ms,
                "codec_degradation_score": result.quality.codec_degradation_score,
            },
            metadata={
                "prediction": result.prediction,
                "device": result.device,
                "risk_score": float(result.risk.risk_score),
                "risk_level": result.risk.risk_level,
                "risk_status": result.risk.status,
                "primary_contribution": float(
                    result.risk.primary_contribution
                ),
                "prosody_contribution": float(
                    result.risk.prosody_contribution
                ),
                "adjusted_spoof_probability": float(
                    result.risk.adjusted_spoof_probability
                ),
                "reasons": list(result.risk.reasons),
            },
        )