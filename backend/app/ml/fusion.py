from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.ml.connectors.base import MLModelOutput


@dataclass(frozen=True)
class FusedMLResult:
    spoof_probability: Optional[float]
    bonafide_probability: Optional[float]
    confidence: float
    models_used: int
    successful_models: int
    status: str
    evidence: dict[str, Any]
    model_outputs: list[dict[str, Any]]


class MultiModelFusion:

    def fuse(
        self,
        outputs: list[MLModelOutput],
    ) -> FusedMLResult:

        if not outputs:
            return FusedMLResult(
                spoof_probability=None,
                bonafide_probability=None,
                confidence=0.0,
                models_used=0,
                successful_models=0,
                status="no_models",
                evidence={},
                model_outputs=[],
            )

        valid = [
            output
            for output in outputs
            if output.status == "ok"
        ]

        if not valid:
            return FusedMLResult(
                spoof_probability=None,
                bonafide_probability=None,
                confidence=0.0,
                models_used=len(outputs),
                successful_models=0,
                status="all_models_failed",
                evidence={},
                model_outputs=[
                    self._serialize(output)
                    for output in outputs
                ],
            )

        spoof_probability = sum(
            output.spoof_probability
            for output in valid
        ) / len(valid)

        bonafide_probability = 1.0 - spoof_probability

        confidence = sum(
            output.confidence
            for output in valid
        ) / len(valid)

        evidence: dict[str, Any] = {}

        for output in valid:
            evidence[output.model_id] = output.evidence

        status = (
            "ok"
            if len(valid) == len(outputs)
            else "partial"
        )

        return FusedMLResult(
            spoof_probability=float(spoof_probability),
            bonafide_probability=float(bonafide_probability),
            confidence=float(confidence),
            models_used=len(outputs),
            successful_models=len(valid),
            status=status,
            evidence=evidence,
            model_outputs=[
                self._serialize(output)
                for output in outputs
            ],
        )

    @staticmethod
    def _serialize(
        output: MLModelOutput,
    ) -> dict[str, Any]:

        return {
            "model_id": output.model_id,
            "model_version": output.model_version,
            "spoof_probability": output.spoof_probability,
            "bonafide_probability": output.bonafide_probability,
            "confidence": output.confidence,
            "latency_ms": output.latency_ms,
            "status": output.status,
            "evidence": output.evidence,
            "quality": output.quality,
            "metadata": output.metadata,
        }
