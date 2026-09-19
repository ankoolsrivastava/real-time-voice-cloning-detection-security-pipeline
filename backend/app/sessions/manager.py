from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.audio.stream_processor import AudioStreamProcessor
from app.audio.types import AudioChunk
from app.temporal.accumulator import TemporalAccumulator


@dataclass
class SessionState:
    session_id: str
    created_at: str
    status: str = "active"
    processor: AudioStreamProcessor = field(
        default_factory=AudioStreamProcessor
    )

    windows_processed: int = 0
    latest_result: Optional[dict] = None
    temporal: TemporalAccumulator = field(
        default_factory=TemporalAccumulator
    )

    def telemetry(self) -> dict:
        return self.processor.telemetry()

    def add_chunk(self, chunk: AudioChunk) -> Optional[dict]:
        processed = self.processor.add_chunk(chunk)

        if processed is None:
            return None

        self.windows_processed += 1

        result = processed.result

        output = {
            "session_id": self.session_id,
            "window": {
                "sequence_start": processed.window.sequence_start,
                "sequence_end": processed.window.sequence_end,
                "timestamp_start": processed.window.timestamp_start,
                "timestamp_end": processed.window.timestamp_end,
                "sample_rate": processed.window.sample_rate,
                "samples": len(processed.window.audio),
            },
            "prediction": result.prediction_label,
            "spoof_probability": result.spoof_probability,
            "bonafide_probability": result.bonafide_probability,
            "prosody_probability": result.prosody_probability,
            "prosody_reliability": result.prosody_reliability,
            "quality": {
                "score": result.quality.quality_score,
                "confidence_multiplier": result.quality.confidence_multiplier,
                "status": result.quality.status,
                "rms_db": result.quality.rms_db,
                "peak": result.quality.peak,
                "clipping_ratio": result.quality.clipping_ratio,
                "duration_sec": result.quality.duration_sec,
                "active_ratio": result.quality.active_ratio,
            },
            "risk": {
                "score": result.risk.risk_score,
                "level": result.risk.risk_level,
                "adjusted_spoof_probability": (
                    result.risk.adjusted_spoof_probability
                ),
                "evidence_confidence": result.risk.evidence_confidence,
                "status": result.risk.status,
                "reasons": result.risk.reasons,
            },
            "model": {
                "version": result.model_version,
                "device": result.device,
            },
            "models": processed.fused.model_outputs,
            "fusion": {
                "spoof_probability": processed.fused.spoof_probability,
                "bonafide_probability": processed.fused.bonafide_probability,
                "confidence": processed.fused.confidence,
                "models_used": processed.fused.models_used,
                "successful_models": processed.fused.successful_models,
                "status": processed.fused.status,
            },
        }

        temporal_result = self.temporal.update(
            result.risk.risk_score
        )
        output["temporal"] = temporal_result.to_dict()

        self.latest_result = output
        return output


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self) -> SessionState:
        session_id = uuid4().hex

        session = SessionState(
            session_id=session_id,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(
                f"Session '{session_id}' not found."
            )

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(
            session_id,
            None,
        ) is not None

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "status": session.status,
                "windows_processed": session.windows_processed,
                "latest_result": session.latest_result,
            }
            for session in self._sessions.values()
        ]

    def reset(self, session_id: str) -> None:
        session = self.get(session_id)
        session.processor.reset()
        session.temporal.reset()
        session.windows_processed = 0
        session.latest_result = None


session_manager = SessionManager()
