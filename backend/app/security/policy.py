from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityDecision:
    action: str
    severity: str
    requires_verification: bool
    message: str


class SecurityPolicy:
    """
    Backend security policy.

    ML produces risk.
    This layer converts risk into an operational action.
    """

    def __init__(
        self,
        medium_threshold: float = 30.0,
        high_threshold: float = 70.0,
        critical_threshold: float = 90.0,
    ) -> None:
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold

    def evaluate(
        self,
        risk_score: float,
        risk_level: str,
        risk_status: str,
    ) -> SecurityDecision:

        if risk_status == "LOW_CONFIDENCE":
            return SecurityDecision(
                action="VERIFY",
                severity="LOW_CONFIDENCE",
                requires_verification=True,
                message=(
                    "Audio quality or evidence confidence is insufficient; "
                    "perform secondary verification."
                ),
            )

        if risk_score >= self.critical_threshold:
            return SecurityDecision(
                action="BLOCK_ESCALATE",
                severity="CRITICAL",
                requires_verification=True,
                message=(
                    "Critical impersonation risk; block or escalate "
                    "according to organizational policy."
                ),
            )

        if risk_score >= self.high_threshold:
            return SecurityDecision(
                action="MFA_CALLBACK_ESCALATE",
                severity="HIGH",
                requires_verification=True,
                message=(
                    "High impersonation risk; require secondary verification."
                ),
            )

        if risk_score >= self.medium_threshold:
            return SecurityDecision(
                action="VERIFY",
                severity="MEDIUM",
                requires_verification=True,
                message=(
                    "Moderate impersonation risk; perform secondary verification."
                ),
            )

        return SecurityDecision(
            action="ALLOW",
            severity="LOW",
            requires_verification=False,
            message="Low impersonation risk.",
        )
