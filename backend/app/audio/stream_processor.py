from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app import config
from app.ml.runtime import MLRuntimeResult, get_ml_runtime
from app.ml.connectors import MLModelRegistry, V2ModelConnector
from app.ml.manager import MultiModelManager
from app.ml.fusion import FusedMLResult

from .chunk_buffer import ChunkBuffer
from .types import AudioChunk, AudioWindow
from .window_manager import WindowManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedWindow:
    window: AudioWindow
    result: MLRuntimeResult
    ml_outputs: list
    fused: FusedMLResult


class AudioStreamProcessor:

    def __init__(self) -> None:
        self.chunk_buffer = ChunkBuffer()
        self.window_manager = WindowManager()

        self.ml_runtime = get_ml_runtime()

        self.ml_registry = MLModelRegistry()

        self.v2_connector = V2ModelConnector()
        self.ml_registry.register(self.v2_connector)

        self.ml_manager = MultiModelManager(self.ml_registry)

    def add_chunk(self, chunk: AudioChunk) -> Optional[ProcessedWindow]:

        if chunk.sample_rate != config.SAMPLE_RATE:
            raise ValueError(
                f"Expected {config.SAMPLE_RATE} Hz audio, "
                f"received {chunk.sample_rate} Hz."
            )

        accepted = self.chunk_buffer.add(chunk)

        if not accepted:
            return None

        self.window_manager.add_chunk(chunk)

        if not self.window_manager.has_window():
            return None

        window = self.window_manager.build_window(
            packet_loss_ratio=self.chunk_buffer.packet_loss_ratio,
            jitter_ms=chunk.jitter_ms,
        )

        if window is None:
            return None

        # ---------------------------------------------------------
        # Run frozen V2 exactly ONCE.
        # ---------------------------------------------------------
        start = time.perf_counter()

        result = self.ml_runtime.predict_waveform(
            waveform=window.audio,
            sample_rate=window.sample_rate,
            packet_loss_ratio=window.packet_loss_ratio,
            jitter_ms=window.jitter_ms,
        )

        v2_latency_ms = (time.perf_counter() - start) * 1000.0

        # Convert the already-computed native V2 result into the
        # generic connector output. No second neural inference.
        v2_output = self.v2_connector.from_runtime_result(
            result=result,
            latency_ms=v2_latency_ms,
        )

        # Future connectors run normally. V2 is supplied as
        # precomputed output.
        ml_outputs, fused_result = self.ml_manager.predict_and_fuse(
            waveform=window.audio,
            sample_rate=window.sample_rate,
            precomputed={
                self.v2_connector.model_id: v2_output,
            },
        )

        logger.info("ML connector results: models=%d", len(ml_outputs))
        for output in ml_outputs:
            logger.debug(
                "ML model=%s version=%s spoof=%.6f bonafide=%.6f "
                "confidence=%.6f latency_ms=%.2f status=%s",
                output.model_id,
                output.model_version,
                output.spoof_probability,
                output.bonafide_probability,
                output.confidence,
                output.latency_ms,
                output.status,
            )

        logger.info(
            "Fused ML result: models=%d successful=%d spoof=%.6f "
            "bonafide=%.6f confidence=%.6f status=%s",
            fused_result.models_used,
            fused_result.successful_models,
            fused_result.spoof_probability,
            fused_result.bonafide_probability,
            fused_result.confidence,
            fused_result.status,
        )

        return ProcessedWindow(
            window=window,
            result=result,
            ml_outputs=ml_outputs,
            fused=fused_result,
        )

    def telemetry(self) -> dict:
        return {
            "buffer": self.chunk_buffer.telemetry(),
            "buffered_samples": self.window_manager.buffered_samples,
            "buffered_duration_sec": (
                self.window_manager.buffered_duration_sec
            ),
        }

    def reset(self) -> None:
        self.chunk_buffer = ChunkBuffer()
        self.window_manager.reset()
