"""
VoiceGuard Dynamic Risk Engine V2

Active evidence fusion:
- frozen V2 primary detector
- deployable prosody scorer
- audio/evidence confidence

Quality confidence attenuates BOTH evidence sources toward neutral.
Quality is never spoof evidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import math

@dataclass
class RiskInput:
    spoof_probability: float
    quality_confidence_multiplier: float = 1.0
    prosody_probability: Optional[float] = None
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
    """Active V2 evidence fusion. Defaults match the frozen 83/17 configuration."""

    def __init__(
        self,
        threshold: float = 0.25,
        primary_weight: float = 0.83,
        prosody_weight: float = 0.17,
        low_risk: float = 30.0,
        high_risk: float = 70.0,
        critical_risk: float = 90.0,
    ):
        if not math.isclose(primary_weight + prosody_weight, 1.0, abs_tol=1e-9):
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.primary_weight = primary_weight
        self.prosody_weight = prosody_weight
        self.low_risk = low_risk
        self.high_risk = high_risk
        self.critical_risk = critical_risk

    @staticmethod
    def _clip(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _attenuate_to_neutral(probability, confidence):
        return 0.5 + (probability - 0.5) * confidence

    def evaluate(self, data: RiskInput) -> RiskOutput:
        quality = self._clip(data.quality_confidence_multiplier)
        evidence = self._clip(data.evidence_confidence)
        confidence = quality * evidence

        primary_raw = self._clip(data.spoof_probability)
        primary_adj = self._attenuate_to_neutral(primary_raw, confidence)
        primary_score = primary_adj * 100.0

        reasons = []
        if data.prosody_probability is not None and data.prosody_reliability > 0:
            prosody_raw = self._clip(data.prosody_probability)
            reliability = self._clip(data.prosody_reliability)
            prosody_adj = self._attenuate_to_neutral(prosody_raw, confidence)
            # Reliability controls secondary influence, while quality already
            # attenuates its evidential strength toward neutral.
            prosody_effective = 0.5 + (prosody_adj - 0.5) * reliability
            prosody_score = prosody_effective * 100.0
            risk = (
                self.primary_weight * primary_score
                + self.prosody_weight * prosody_score
            )
            reasons.append("primary V2 + deployable prosody evidence")
        else:
            prosody_score = 50.0
            risk = primary_score
            reasons.append("primary V2 only; prosody unavailable")

        risk = self._clip(risk, 0.0, 100.0)

        if risk >= self.critical_risk:
            level = "CRITICAL"
        elif risk >= self.high_risk:
            level = "HIGH"
        elif risk >= self.low_risk:
            level = "MEDIUM"
        else:
            level = "LOW"

        status = "USABLE"
        if quality < 0.50 or evidence < 0.50:
            status = "LOW_CONFIDENCE"
            reasons.append("degraded confidence: collect more reliable windows")

        reasons.append(
            f"primary_decision={'spoof' if primary_raw >= self.threshold else 'bonafide'} "
            f"at frozen threshold {self.threshold:.2f}"
        )

        return RiskOutput(
            risk_score=round(risk, 4),
            risk_level=level,
            adjusted_spoof_probability=round(primary_adj, 6),
            primary_contribution=round(primary_score, 4),
            prosody_contribution=round(prosody_score, 4),
            evidence_confidence=round(confidence, 6),
            status=status,
            reasons=reasons,
        )
