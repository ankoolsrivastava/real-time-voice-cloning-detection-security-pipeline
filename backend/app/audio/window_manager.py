from __future__ import annotations

from collections import deque

import numpy as np

from app import config

from .types import AudioChunk, AudioWindow


class WindowManager:
    """
    Constructs exact frozen-V2 inference windows.

    Required V2 window:
        sample rate = 16000 Hz
        channels    = mono
        samples     = 160000
        duration    = 10 seconds
    """

    def __init__(self) -> None:
        self.sample_rate = config.SAMPLE_RATE
        self.window_samples = config.WINDOW_SAMPLES

        self._chunks: deque[AudioChunk] = deque()

        self._total_samples = 0

        self._window_sequence_start: int | None = None
        self._window_timestamp_start: float | None = None

    def add_chunk(self, chunk: AudioChunk) -> None:
        if chunk.sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz audio, "
                f"received {chunk.sample_rate} Hz"
            )

        if self._window_sequence_start is None:
            self._window_sequence_start = chunk.sequence_number
            self._window_timestamp_start = chunk.timestamp_start

        self._chunks.append(chunk)
        self._total_samples += len(chunk.audio)

    def has_window(self) -> bool:
        return self._total_samples >= self.window_samples

    def build_window(
        self,
        packet_loss_ratio: float = 0.0,
        jitter_ms: float | None = None,
    ) -> AudioWindow | None:

        if not self.has_window():
            return None

        samples_needed = self.window_samples

        pieces: list[np.ndarray] = []

        sequence_start = self._window_sequence_start
        timestamp_start = self._window_timestamp_start

        sequence_end = sequence_start
        timestamp_end = timestamp_start

        while samples_needed > 0 and self._chunks:

            chunk = self._chunks[0]
            audio = np.asarray(
                chunk.audio,
                dtype=np.float32,
            )

            take = min(
                samples_needed,
                len(audio),
            )

            pieces.append(audio[:take])

            samples_needed -= take

            sequence_end = chunk.sequence_number
            timestamp_end = (
                chunk.timestamp_start
                + (take / self.sample_rate)
            )

            if take == len(audio):
                self._chunks.popleft()
            else:
                remaining = audio[take:]

                replacement = AudioChunk(
                    audio=remaining,
                    sample_rate=chunk.sample_rate,
                    sequence_number=chunk.sequence_number,
                    timestamp_start=timestamp_end,
                    timestamp_end=chunk.timestamp_end,
                    received_at=chunk.received_at,
                    packet_loss_before=chunk.packet_loss_before,
                    jitter_ms=chunk.jitter_ms,
                )

                self._chunks[0] = replacement

            self._total_samples -= take

        if samples_needed != 0:
            raise RuntimeError(
                "WindowManager failed to construct "
                "the required exact V2 window."
            )

        waveform = np.concatenate(pieces).astype(
            np.float32,
            copy=False,
        )

        if len(waveform) != self.window_samples:
            raise RuntimeError(
                f"Constructed {len(waveform)} samples; "
                f"expected {self.window_samples}."
            )

        window = AudioWindow(
            audio=waveform,
            sample_rate=self.sample_rate,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            packet_loss_ratio=packet_loss_ratio,
            jitter_ms=jitter_ms,
        )

        if self._chunks:
            self._window_sequence_start = (
                self._chunks[0].sequence_number
            )
            self._window_timestamp_start = (
                self._chunks[0].timestamp_start
            )
        else:
            self._window_sequence_start = None
            self._window_timestamp_start = None

        return window

    @property
    def buffered_samples(self) -> int:
        return self._total_samples

    @property
    def buffered_duration_sec(self) -> float:
        return self._total_samples / self.sample_rate

    def reset(self) -> None:
        self._chunks.clear()
        self._total_samples = 0
        self._window_sequence_start = None
        self._window_timestamp_start = None
