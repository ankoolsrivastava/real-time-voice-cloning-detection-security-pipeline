from __future__ import annotations

from app.security.policy import SecurityPolicy
from app.sessions.manager import SessionManager


class DetectionService:
    def __init__(
        self,
        sessions: SessionManager,
        security_policy: SecurityPolicy,
    ) -> None:
        self.sessions = sessions
        self.security_policy = security_policy

    def process_result(
        self,
        session_id: str,
        result: dict,
    ) -> dict:

        risk = result["risk"]

        decision = self.security_policy.evaluate(
            risk_score=float(risk["score"]),
            risk_level=str(risk["level"]),
            risk_status=str(risk["status"]),
        )

        result["security"] = {
            "action": decision.action,
            "severity": decision.severity,
            "requires_verification": (
                decision.requires_verification
            ),
            "message": decision.message,
        }

        return result


detection_service = DetectionService(
    sessions=__import__(
        "app.sessions.manager",
        fromlist=["session_manager"],
    ).session_manager,
    security_policy=SecurityPolicy(),
)
