from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class TemporalResult:
    current_risk_score: float
    accumulated_risk_score: float
    max_risk_score: float
    trend: str
    windows_seen: int
    high_risk_windows: int
    critical_risk_windows: int
    persistent_high_risk: bool
    persistent_critical_risk: bool
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalAccumulator:
    """Accumulates window-level ML risk without replacing the frozen ML risk engine."""

    def __init__(
        self,
        max_history: int = 6,
        high_threshold: float = 70.0,
        critical_threshold: float = 90.0,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")

        self.max_history = max_history
        self.high_threshold = float(high_threshold)
        self.critical_threshold = float(critical_threshold)
        self._scores: deque[float] = deque(maxlen=max_history)

    def reset(self) -> None:
        self._scores.clear()

    @property
    def windows_seen(self) -> int:
        return len(self._scores)

    def update(self, risk_score: float) -> TemporalResult:
        score = float(risk_score)

        if not 0.0 <= score <= 100.0:
            raise ValueError("risk_score must be between 0 and 100")

        self._scores.append(score)

        scores = list(self._scores)
        accumulated = sum(scores) / len(scores)
        maximum = max(scores)

        if len(scores) < 2:
            trend = "stable"
        else:
            delta = scores[-1] - scores[-2]
            if delta > 5.0:
                trend = "rising"
            elif delta < -5.0:
                trend = "falling"
            else:
                trend = "stable"

        high_count = sum(s >= self.high_threshold for s in scores)
        critical_count = sum(s >= self.critical_threshold for s in scores)

        return TemporalResult(
            current_risk_score=score,
            accumulated_risk_score=round(accumulated, 4),
            max_risk_score=round(maximum, 4),
            trend=trend,
            windows_seen=len(scores),
            high_risk_windows=high_count,
            critical_risk_windows=critical_count,
            persistent_high_risk=high_count >= 2,
            persistent_critical_risk=critical_count >= 2,
        )
