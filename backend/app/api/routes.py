from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import MODEL_PATH, ML_HANDOFF_ROOT, SAMPLE_RATE, WINDOW_DURATION_SEC
from app.sessions.manager import session_manager
from app.services.detection_service import detection_service

router = APIRouter()


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    model_version: str


class AudioChunkRequest(BaseModel):
    sequence_number: int = Field(ge=0)
    timestamp_start: float = Field(ge=0)
    timestamp_end: float = Field(ge=0)
    received_at: float = Field(ge=0)
    sample_rate: int = Field(default=16000, gt=0)
    audio: list[float]
    packet_loss_before: int = Field(default=0, ge=0)
    jitter_ms: float | None = Field(default=None, ge=0)


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "VoiceGuard Backend",
    }


@router.get("/model")
def model_info() -> dict:
    return {
        "model_version": "voiceguard_v2_epoch8",
        "checkpoint": str(MODEL_PATH),
        "handoff_root": str(ML_HANDOFF_ROOT),
        "status": "loaded",
    }


@router.get("/integration")
def integration_contract() -> dict:
    return {
        "service": "VoiceGuard",
        "api_version": "v1",
        "rest_base": "/api",
        "websocket_path": "/api/ws/{session_id}",
        "sample_rate": SAMPLE_RATE,
        "chunk_seconds": 1,
        "window_seconds": WINDOW_DURATION_SEC,
        "model_version": "voiceguard_v2_epoch8",
        "capabilities": {
            "live_streaming": True,
            "risk_scoring": True,
            "prosody": True,
            "acoustic_spectral": True,
            "temporal": True,
            "security_policy": True,
            "multi_model": False,
            "speaker_consistency": False,
        },
    }


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
)
def create_session() -> CreateSessionResponse:
    session = session_manager.create()

    return CreateSessionResponse(
        session_id=session.session_id,
        status=session.status,
        model_version="voiceguard_v2_epoch8",
    )


@router.get("/sessions")
def list_sessions() -> list[dict]:
    return session_manager.list_sessions()


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        session = session_manager.get(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "status": session.status,
        "windows_processed": session.windows_processed,
        "latest_result": session.latest_result,
        "telemetry": session.processor.telemetry(),
    }


@router.post("/sessions/{session_id}/chunks")
def add_chunk(
    session_id: str,
    request: AudioChunkRequest,
) -> dict:

    try:
        session = session_manager.get(session_id)

        import numpy as np
        from app.audio.types import AudioChunk

        chunk = AudioChunk(
            audio=np.asarray(
                request.audio,
                dtype=np.float32,
            ),
            sample_rate=request.sample_rate,
            sequence_number=request.sequence_number,
            timestamp_start=request.timestamp_start,
            timestamp_end=request.timestamp_end,
            received_at=request.received_at,
            packet_loss_before=request.packet_loss_before,
            jitter_ms=request.jitter_ms,
        )

        result = session.add_chunk(chunk)

        if result is None:
            return {
                "status": "buffering",
                "window_ready": False,
                "telemetry": session.processor.telemetry(),
            }

        return detection_service.process_result(
            session_id=session_id,
            result=result,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    try:
        session_manager.reset(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    return {
        "session_id": session_id,
        "status": "reset",
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if not session_manager.delete(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    return {
        "session_id": session_id,
        "status": "deleted",
    }
