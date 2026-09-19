from app.ml.connectors import MLModelRegistry, V2ModelConnector
from app.ml.manager import MultiModelManager
from app.ml.fusion import MultiModelFusion


def test_v2_connector_registration():
    registry = MLModelRegistry()
    registry.register(V2ModelConnector())

    assert registry.ids() == ["voiceguard_v2"]
    assert len(registry) == 1


def test_fusion_empty():
    result = MultiModelFusion().fuse([])

    assert result.status == "no_models"
    assert result.models_used == 0
    assert result.successful_models == 0
    assert result.confidence == 0.0


def test_fusion_single_model():
    connector = V2ModelConnector()

    registry = MLModelRegistry()
    registry.register(connector)

    manager = MultiModelManager(registry)

    assert manager.model_ids() == ["voiceguard_v2"]
    assert manager.model_count() == 1
