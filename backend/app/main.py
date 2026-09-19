from __future__ import annotations

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.api.websocket import voice_stream

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

_console = logging.StreamHandler()
_console.setFormatter(_formatter)

_file = RotatingFileHandler(
    LOG_DIR / "voiceguard.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file.setFormatter(_formatter)

_root = logging.getLogger()
_root.setLevel(logging.INFO)

if not _root.handlers:
    _root.addHandler(_console)
    _root.addHandler(_file)
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ml.runtime import get_ml_runtime
from app.api.routes import (
    health,
    model_info,
    create_session,
    list_sessions,
    get_session,
    add_chunk,
    reset_session,
    delete_session,
    integration_contract,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ml_runtime()
    yield


app = FastAPI(
    title="VoiceGuard Backend",
    version="1.0.0",
    description="Real-time voice integrity and spoof detection backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_api_route("/api/health", health, methods=["GET"], response_model=dict)
app.add_api_route("/api/model", model_info, methods=["GET"], response_model=dict)
app.add_api_route("/api/integration", integration_contract, methods=["GET"], response_model=dict)

app.add_api_route(
    "/api/sessions",
    create_session,
    methods=["POST"],
)

app.add_api_route(
    "/api/sessions",
    list_sessions,
    methods=["GET"],
)

app.add_api_route(
    "/api/sessions/{session_id}",
    get_session,
    methods=["GET"],
)

app.add_api_route(
    "/api/sessions/{session_id}/chunks",
    add_chunk,
    methods=["POST"],
)

app.add_api_route(
    "/api/sessions/{session_id}/reset",
    reset_session,
    methods=["POST"],
)

app.add_api_route(
    "/api/sessions/{session_id}",
    delete_session,
    methods=["DELETE"],
)
app.add_api_websocket_route(
    "/api/ws/{session_id}",
    voice_stream,
)

