import time

import numpy as np

from app.audio.types import AudioChunk
from app.audio.chunk_buffer import ChunkBuffer
from app.audio.window_manager import WindowManager


def make_chunk(samples: int) -> AudioChunk:
    return AudioChunk(
        audio=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
        sequence_number=0,
        timestamp_start=0.0,
        timestamp_end=samples / 16000,
        received_at=time.time(),
    )


def test_chunk_buffer_accepts_ordered_chunk():
    buffer = ChunkBuffer()

    chunk = make_chunk(1600)

    assert buffer.add(chunk) is True
    assert len(buffer) == 1


def test_window_manager_builds_exact_10_second_window():
    manager = WindowManager()

    chunk = make_chunk(160000)

    manager.add_chunk(chunk)

    assert manager.has_window() is True

    window = manager.build_window(
        packet_loss_ratio=0.0,
        jitter_ms=0.0,
    )

    assert window is not None
    assert window.sample_rate == 16000
    assert len(window.audio) == 160000

