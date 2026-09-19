from __future__ import annotations

import json

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.audio.types import AudioChunk
from app.sessions.manager import session_manager
from app.services.detection_service import detection_service


async def voice_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    try:
        session = session_manager.get(session_id)

        await websocket.send_json({
            "status": "connected",
            "session_id": session_id,
        })

        while True:
            try:
                message = await websocket.receive_text()
                payload = json.loads(message)

                required = (
                    "audio",
                    "sequence_number",
                    "timestamp_start",
                    "timestamp_end",
                )

                missing = [
                    key for key in required
                    if key not in payload
                ]

                if missing:
                    await websocket.send_json({
                        "status": "error",
                        "error": "missing_fields",
                        "fields": missing,
                    })
                    continue

                audio = np.asarray(
                    payload["audio"],
                    dtype=np.float32,
                )

                if audio.ndim != 1 or audio.size == 0:
                    await websocket.send_json({
                        "status": "error",
                        "error": "invalid_audio",
                    })
                    continue

                chunk = AudioChunk(
                    audio=audio,
                    sample_rate=int(
                        payload.get("sample_rate", 16000)
                    ),
                    sequence_number=int(
                        payload["sequence_number"]
                    ),
                    timestamp_start=float(
                        payload["timestamp_start"]
                    ),
                    timestamp_end=float(
                        payload["timestamp_end"]
                    ),
                    received_at=float(
                        payload.get("received_at", 0.0)
                    ),
                    packet_loss_before=int(
                        payload.get("packet_loss_before", 0)
                    ),
                    jitter_ms=(
                        None
                        if payload.get("jitter_ms") is None
                        else float(payload["jitter_ms"])
                    ),
                )

                result = session.add_chunk(chunk)

                if result is None:
                    await websocket.send_json({
                        "status": "buffering",
                        "window_ready": False,
                        "telemetry": session.processor.telemetry(),
                    })
                    continue

                response = detection_service.process_result(
                    session_id=session_id,
                    result=result,
                )

                await websocket.send_json(response)

            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                await websocket.send_json({
                    "status": "error",
                    "error": "invalid_request",
                    "detail": str(exc),
                })

    except WebSocketDisconnect:
        return

    except KeyError:
        try:
            await websocket.send_json({
                "status": "error",
                "error": "session_not_found",
            })
            await websocket.close(code=1008)
        except Exception:
            pass

    except Exception as exc:
        try:
            await websocket.send_json({
                "status": "error",
                "error": "internal_error",
                "detail": str(exc),
            })
        except Exception:
            pass

