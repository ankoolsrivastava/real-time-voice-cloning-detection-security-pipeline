"""
VoiceGuard Dynamic Risk Engine V1
---------------------------------
Transparent evidence-fusion layer above the frozen V2 detector.

Important:
- This is NOT a retrained ML model.
- Audio/network degradation reduces confidence; it is never spoof evidence.
- Speaker consistency is not used because V1 was rejected for deployment.
- Prosody is secondary evidence and can be unavailable.
- The engine is designed for window-by-window near-live inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import math


@dataclass
class RiskInput:
    spoof_probability: float
    quality_confidence_multiplier: float = 1.0
    prosody_evidence: Optional[float] = None
    prosody_reliability: float = 0.0
    evidence_confidence: float = 1.0


@dataclass
class RiskOutput:
    risk_score: float
    risk_level: str
    adjusted_spoof_probability: float
    primary_contribution: float
    prosody_contribution: float
    evidence_confidence: float
    status: str
    reasons: List[str] = field(default_factory=list)


class VoiceGuardRiskEngine:
    """
    V1 transparent risk/evidence fusion.

    Primary detector:
        spoof_probability in [0,1]

    Prosody:
        evidence in [-1,1]
        -1 = evidence toward bonafide
         0 = neutral
        +1 = evidence toward spoof

    Quality:
        confidence multiplier in [0,1].
        It shrinks evidence toward neutral rather than creating spoof evidence.

    evidence_confidence:
        additional confidence factor in [0,1], supplied by the evidence layer.

    The output is a 0-100 risk score.
    """

    def __init__(
        self,
        threshold: float = 0.25,
        primary_weight: float = 0.89,
        prosody_weight: float = 0.11,
        high_risk: float = 70.0,
        critical_risk: float = 90.0,
        low_risk: float = 30.0,
    ):
        if not math.isclose(primary_weight + prosody_weight, 1.0, abs_tol=1e-9):
            raise ValueError("primary_weight + prosody_weight must equal 1.0")
        self.threshold = threshold
        self.primary_weight = primary_weight
        self.prosody_weight = prosody_weight
        self.high_risk = high_risk
        self.critical_risk = critical_risk
        self.low_risk = low_risk

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _adjust_probability(self, p: float, confidence: float) -> float:
        # Degraded quality/confidence moves the probability toward 0.5.
        return 0.5 + (p - 0.5) * confidence

    def evaluate(self, data: RiskInput) -> RiskOutput:
        p = self._clip(float(data.spoof_probability), 0.0, 1.0)
        quality = self._clip(float(data.quality_confidence_multiplier), 0.0, 1.0)
        evidence_conf = self._clip(float(data.evidence_confidence), 0.0, 1.0)

        adjusted_p = self._adjust_probability(p, quality * evidence_conf)

        # Primary detector contribution is represented on a 0-100 risk scale.
        primary = adjusted_p * 100.0

        prosody_contribution = 50.0
        reasons: List[str] = []

        if data.prosody_evidence is not None and data.prosody_reliability > 0:
            pe = self._clip(float(data.prosody_evidence), -1.0, 1.0)
            pr = self._clip(float(data.prosody_reliability), 0.0, 1.0)

            # Convert [-1,1] to [0,100], then attenuate by reliability.
            raw_prosody = 50.0 + 50.0 * pe
            prosody_contribution = 50.0 + (raw_prosody - 50.0) * pr

            fused = (
                self.primary_weight * primary
                + self.prosody_weight * prosody_contribution
            )

            reasons.append("primary detector + secondary prosody evidence")
        else:
            # No prosody: do not fabricate evidence. Renormalize to primary only.
            fused = primary
            reasons.append("primary detector only; prosody unavailable")

        # Final confidence attenuation is already applied to primary evidence.
        # Never push a degraded signal upward merely because quality is poor.
        risk = self._clip(fused, 0.0, 100.0)

        if risk >= self.critical_risk:
            level = "CRITICAL"
        elif risk >= self.high_risk:
            level = "HIGH"
        elif risk >= self.low_risk:
            level = "MEDIUM"
        else:
            level = "LOW"

        if quality < 0.50:
            status = "LOW_CONFIDENCE"
            reasons.append("audio quality degraded: collect more reliable windows")
        elif evidence_conf < 0.50:
            status = "LOW_CONFIDENCE"
            reasons.append("overall evidence confidence degraded")
        else:
            status = "USABLE"

        # Threshold is retained as an explicit model decision reference.
        primary_decision = p >= self.threshold
        reasons.append(
            "primary_decision="
            + ("spoof" if primary_decision else "bonafide")
            + f" at frozen threshold {self.threshold:.2f}"
        )

        return RiskOutput(
            risk_score=round(risk, 4),
            risk_level=level,
            adjusted_spoof_probability=round(adjusted_p, 6),
            primary_contribution=round(primary, 4),
            prosody_contribution=round(prosody_contribution, 4),
            evidence_confidence=round(quality * evidence_conf, 6),
            status=status,
            reasons=reasons,
        )


@dataclass
class EvidenceWindow:
    risk_score: float
    evidence_confidence: float
    usable: bool


class TemporalRiskAccumulator:
    """
    Optional near-live accumulation across reliable windows.

    It deliberately ignores low-confidence windows instead of treating
    packet loss/jitter/compression as spoof evidence.
    """

    def __init__(self, max_windows: int = 8):
        if max_windows < 1:
            raise ValueError("max_windows must be >= 1")
        self.max_windows = max_windows
        self.windows: List[EvidenceWindow] = []

    def add(self, result: RiskOutput) -> None:
        if result.status == "LOW_CONFIDENCE":
            return
        self.windows.append(
            EvidenceWindow(
                risk_score=result.risk_score,
                evidence_confidence=result.evidence_confidence,
                usable=True,
            )
        )
        self.windows = self.windows[-self.max_windows :]

    def aggregate(self) -> Optional[float]:
        if not self.windows:
            return None

        weights = [max(w.evidence_confidence, 1e-6) for w in self.windows]
        scores = [w.risk_score for w in self.windows]
        return round(
            sum(s * w for s, w in zip(scores, weights)) / sum(weights),
            4,
        )
