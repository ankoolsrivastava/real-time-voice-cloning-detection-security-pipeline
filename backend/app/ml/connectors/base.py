from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MLModelOutput:
    model_id: str
    model_version: str

    spoof_probability: float
    bonafide_probability: float
    confidence: float

    latency_ms: float
    status: str = "ok"

    # PS-level evidence exposed by this model.
    evidence: dict[str, Any] = field(default_factory=dict)

    # Model-native quality/telemetry information.
    quality: dict[str, Any] = field(default_factory=dict)

    # Model-specific information that must not be lost.
    metadata: dict[str, Any] = field(default_factory=dict)


class MLModelConnector(ABC):

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict_waveform(
        self,
        waveform,
        sample_rate: int,
    ) -> MLModelOutput:
        raise NotImplementedError
