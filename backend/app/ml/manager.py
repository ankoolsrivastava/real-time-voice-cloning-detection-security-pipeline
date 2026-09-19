from __future__ import annotations

from app.ml.connectors import (
    MLModelConnector,
    MLModelOutput,
    MLModelRegistry,
)
from app.ml.fusion import FusedMLResult, MultiModelFusion


class MultiModelManager:

    def __init__(self, registry: MLModelRegistry):
        self.registry = registry
        self.fusion = MultiModelFusion()

    def predict(
        self,
        waveform,
        sample_rate: int,
        precomputed: dict[str, MLModelOutput] | None = None,
    ) -> list[MLModelOutput]:

        outputs: list[MLModelOutput] = []
        precomputed = precomputed or {}

        for connector in self.registry.all():

            if connector.model_id in precomputed:
                outputs.append(precomputed[connector.model_id])
                continue

            try:
                output = connector.predict_waveform(
                    waveform,
                    sample_rate=sample_rate,
                )
                outputs.append(output)

            except Exception as exc:
                outputs.append(
                    MLModelOutput(
                        model_id=connector.model_id,
                        model_version=connector.model_version,
                        spoof_probability=None,
                        bonafide_probability=None,
                        confidence=0.0,
                        latency_ms=0.0,
                        status="error",
                        evidence={},
                        quality={},
                        metadata={"error": str(exc)},
                    )
                )

        return outputs

    def predict_and_fuse(
        self,
        waveform,
        sample_rate: int,
        precomputed: dict[str, MLModelOutput] | None = None,
    ) -> tuple[list[MLModelOutput], FusedMLResult]:

        outputs = self.predict(
            waveform=waveform,
            sample_rate=sample_rate,
            precomputed=precomputed,
        )

        fused = self.fusion.fuse(outputs)

        return outputs, fused

    def model_ids(self) -> list[str]:
        return self.registry.ids()

    def model_count(self) -> int:
        return len(self.registry)
