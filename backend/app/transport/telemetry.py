from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TransportSnapshot:
    total_received: int
    total_duplicates: int
    total_out_of_order: int
    total_missing: int
    packet_loss_ratio: float
    jitter_ms: Optional[float]


class TransportTelemetry:
    """
    Transport-level telemetry for streamed audio.

    This component does not perform ML inference and does not
    classify audio as spoofed.

    A sequence gap is not immediately considered permanent loss.
    Packets can arrive late/out of order within the reorder window.
    """

    def __init__(self, reorder_window: int = 3) -> None:
        if reorder_window < 0:
            raise ValueError(
                "reorder_window must be non-negative"
            )

        self.reorder_window = reorder_window

        self.total_received = 0
        self.total_duplicates = 0
        self.total_out_of_order = 0
        self.total_missing = 0

        self._highest_sequence: Optional[int] = None
        self._seen: set[int] = set()

        self._previous_timestamp: Optional[float] = None
        self._previous_sequence: Optional[int] = None

        self._jitter_ms: Optional[float] = None

    def record(
        self,
        sequence_number: int,
        timestamp_start: float,
    ) -> None:

        if sequence_number < 0:
            raise ValueError(
                "sequence_number must be non-negative"
            )

        if timestamp_start < 0:
            raise ValueError(
                "timestamp_start must be non-negative"
            )

        self.total_received += 1

        # --------------------------------------------------
        # Duplicate detection
        # --------------------------------------------------
        if sequence_number in self._seen:
            self.total_duplicates += 1
            return

        self._seen.add(sequence_number)

        # --------------------------------------------------
        # Sequence ordering / loss accounting
        # --------------------------------------------------
        if self._highest_sequence is None:
            self._highest_sequence = sequence_number

        elif sequence_number > self._highest_sequence:

            gap = sequence_number - self._highest_sequence - 1

            if gap > 0:
                # Do not immediately treat the gap as permanent.
                # It becomes provisional missing telemetry until
                # the packet falls outside the reorder window.
                self.total_missing += max(
                    0,
                    gap - self.reorder_window,
                )

            self._highest_sequence = sequence_number

        else:
            self.total_out_of_order += 1

        # --------------------------------------------------
        # Jitter estimation
        #
        # Compare observed arrival/timestamp spacing against
        # the expected spacing implied by sequence progression.
        # --------------------------------------------------
        if (
            self._previous_timestamp is not None
            and self._previous_sequence is not None
            and sequence_number > self._previous_sequence
        ):

            sequence_delta = (
                sequence_number
                - self._previous_sequence
            )

            observed_delta = (
                timestamp_start
                - self._previous_timestamp
            )

            if observed_delta >= 0:

                expected_delta = observed_delta / sequence_delta

                deviation = abs(
                    observed_delta
                    - (
                        expected_delta
                        * sequence_delta
                    )
                )

                deviation_ms = deviation * 1000.0

                if self._jitter_ms is None:
                    self._jitter_ms = deviation_ms
                else:
                    # Exponential moving average.
                    alpha = 0.1
                    self._jitter_ms = (
                        (1 - alpha) * self._jitter_ms
                        + alpha * deviation_ms
                    )

        if (
            self._previous_sequence is None
            or sequence_number > self._previous_sequence
        ):
            self._previous_sequence = sequence_number
            self._previous_timestamp = timestamp_start

    @property
    def packet_loss_ratio(self) -> float:
        denominator = (
            self.total_received
            + self.total_missing
        )

        if denominator <= 0:
            return 0.0

        return (
            self.total_missing
            / denominator
        )

    @property
    def jitter_ms(self) -> Optional[float]:
        return self._jitter_ms

    def snapshot(self) -> TransportSnapshot:
        return TransportSnapshot(
            total_received=self.total_received,
            total_duplicates=self.total_duplicates,
            total_out_of_order=self.total_out_of_order,
            total_missing=self.total_missing,
            packet_loss_ratio=self.packet_loss_ratio,
            jitter_ms=self.jitter_ms,
        )

    def as_dict(self) -> dict:
        snapshot = self.snapshot()

        return {
            "total_received": snapshot.total_received,
            "total_duplicates": snapshot.total_duplicates,
            "total_out_of_order": snapshot.total_out_of_order,
            "total_missing": snapshot.total_missing,
            "packet_loss_ratio": snapshot.packet_loss_ratio,
            "jitter_ms": snapshot.jitter_ms,
        }

    def reset(self) -> None:
        self.total_received = 0
        self.total_duplicates = 0
        self.total_out_of_order = 0
        self.total_missing = 0

        self._highest_sequence = None
        self._seen.clear()

        self._previous_timestamp = None
        self._previous_sequence = None

        self._jitter_ms = None
