from __future__ import annotations

from collections import OrderedDict
from typing import List

from .types import AudioChunk


class ChunkBuffer:
    """
    Ordered audio chunk buffer.

    The buffer is transport-aware but ML-agnostic.
    It records missing/duplicate/out-of-order chunks without
    turning transport degradation into spoof evidence.
    """

    def __init__(self, max_chunks: int = 256) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        self.max_chunks = max_chunks

        self._chunks: OrderedDict[int, AudioChunk] = OrderedDict()

        self._expected_sequence: int | None = None

        self.total_chunks_received = 0
        self.total_duplicate_chunks = 0
        self.total_missing_chunks = 0
        self.total_out_of_order_chunks = 0

    def add(self, chunk: AudioChunk) -> bool:
        """
        Add a chunk.

        Returns:
            True if the chunk was accepted.
            False if it was a duplicate.
        """

        self.total_chunks_received += 1

        sequence = chunk.sequence_number

        if sequence in self._chunks:
            self.total_duplicate_chunks += 1
            return False

        if self._expected_sequence is None:
            self._expected_sequence = sequence

        elif sequence < self._expected_sequence:
            self.total_out_of_order_chunks += 1

        elif sequence > self._expected_sequence:
            missing = sequence - self._expected_sequence
            self.total_missing_chunks += missing

        self._chunks[sequence] = chunk

        self._chunks = OrderedDict(
            sorted(self._chunks.items())
        )

        self._expected_sequence = max(
            self._expected_sequence,
            sequence + 1,
        )

        while len(self._chunks) > self.max_chunks:
            self._chunks.popitem(last=False)

        return True

    def ordered_chunks(self) -> List[AudioChunk]:
        return list(self._chunks.values())

    def pop_all(self) -> List[AudioChunk]:
        chunks = self.ordered_chunks()
        self._chunks.clear()
        return chunks

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def packet_loss_ratio(self) -> float:
        denominator = (
            self.total_chunks_received
            + self.total_missing_chunks
        )

        if denominator <= 0:
            return 0.0

        return self.total_missing_chunks / denominator

    def telemetry(self) -> dict:
        return {
            "buffered_chunks": len(self),
            "total_chunks_received": self.total_chunks_received,
            "total_duplicate_chunks": self.total_duplicate_chunks,
            "total_missing_chunks": self.total_missing_chunks,
            "total_out_of_order_chunks": self.total_out_of_order_chunks,
            "packet_loss_ratio": self.packet_loss_ratio,
        }
