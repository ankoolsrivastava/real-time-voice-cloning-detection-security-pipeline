from __future__ import annotations

from .base import MLModelConnector


class MLModelRegistry:
    """Runtime registry for any number of ML model connectors."""

    def __init__(self) -> None:
        self._connectors: dict[str, MLModelConnector] = {}

    def register(self, connector: MLModelConnector) -> None:
        model_id = connector.model_id

        if not model_id:
            raise ValueError("ML connector model_id cannot be empty")

        if model_id in self._connectors:
            raise ValueError(f"ML connector already registered: {model_id}")

        self._connectors[model_id] = connector

    def get(self, model_id: str) -> MLModelConnector:
        try:
            return self._connectors[model_id]
        except KeyError as exc:
            raise KeyError(f"ML connector not registered: {model_id}") from exc

    def all(self) -> list[MLModelConnector]:
        return list(self._connectors.values())

    def ids(self) -> list[str]:
        return list(self._connectors.keys())

    def __len__(self) -> int:
        return len(self._connectors)