from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """
    One ordered piece of incoming audio.

    Audio is expected to be mono float32 at the backend's
    required sample rate.
    """

    audio: np.ndarray
    sample_rate: int
    sequence_number: int
    timestamp_start: float
    timestamp_end: float
    received_at: float

    packet_loss_before: int = 0
    jitter_ms: Optional[float] = None

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio)

        if audio.ndim != 1:
            raise ValueError(
                f"AudioChunk audio must be 1-D, got {audio.shape}"
            )

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")

        if self.timestamp_end < self.timestamp_start:
            raise ValueError(
                "timestamp_end cannot be earlier than timestamp_start"
            )

        if self.packet_loss_before < 0:
            raise ValueError(
                "packet_loss_before cannot be negative"
            )

        if self.jitter_ms is not None and self.jitter_ms < 0:
            raise ValueError(
                "jitter_ms cannot be negative"
            )

        if not np.isfinite(audio).all():
            raise ValueError(
                "AudioChunk contains NaN or Inf"
            )


@dataclass(frozen=True)
class AudioWindow:
    """
    Exact inference window passed to the ML runtime.
    """

    audio: np.ndarray
    sample_rate: int
    sequence_start: int
    sequence_end: int
    timestamp_start: float
    timestamp_end: float

    packet_loss_ratio: float = 0.0
    jitter_ms: Optional[float] = None

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio)

        if audio.ndim != 1:
            raise ValueError(
                f"AudioWindow audio must be 1-D, got {audio.shape}"
            )

        if not np.isfinite(audio).all():
            raise ValueError(
                "AudioWindow contains NaN or Inf"
            )

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if self.sequence_end < self.sequence_start:
            raise ValueError(
                "sequence_end cannot be earlier than sequence_start"
            )

        if not 0.0 <= self.packet_loss_ratio <= 1.0:
            raise ValueError(
                "packet_loss_ratio must be between 0 and 1"
            )

        if self.jitter_ms is not None and self.jitter_ms < 0:
            raise ValueError(
                "jitter_ms cannot be negative"
            )
