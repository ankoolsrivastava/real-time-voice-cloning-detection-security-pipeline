from voiceguard_dynamic_risk_engine_v1 import (
    RiskInput,
    VoiceGuardRiskEngine,
    TemporalRiskAccumulator,
)


def test_bonafide_good_quality():
    e = VoiceGuardRiskEngine()
    r = e.evaluate(RiskInput(spoof_probability=0.05))
    assert r.risk_score < 30
    assert r.risk_level == "LOW"


def test_spoof_good_quality():
    e = VoiceGuardRiskEngine()
    r = e.evaluate(RiskInput(spoof_probability=0.95))
    assert r.risk_score > 90
    assert r.risk_level == "CRITICAL"


def test_bad_quality_moves_toward_neutral():
    e = VoiceGuardRiskEngine()
    good = e.evaluate(RiskInput(spoof_probability=0.95, quality_confidence_multiplier=1.0))
    poor = e.evaluate(RiskInput(spoof_probability=0.95, quality_confidence_multiplier=0.1))
    assert poor.risk_score < good.risk_score
    assert poor.status == "LOW_CONFIDENCE"


def test_poor_quality_does_not_create_spoof_evidence():
    e = VoiceGuardRiskEngine()
    r = e.evaluate(
        RiskInput(spoof_probability=0.10, quality_confidence_multiplier=0.0)
    )
    assert r.risk_score <= 50
    assert r.status == "LOW_CONFIDENCE"


def test_prosody_is_secondary():
    e = VoiceGuardRiskEngine()
    primary_only = e.evaluate(RiskInput(spoof_probability=0.60))
    with_prosody = e.evaluate(
        RiskInput(
            spoof_probability=0.60,
            prosody_evidence=1.0,
            prosody_reliability=1.0,
        )
    )
    assert with_prosody.risk_score > primary_only.risk_score


def test_temporal_accumulator_ignores_low_confidence():
    e = VoiceGuardRiskEngine()
    acc = TemporalRiskAccumulator(max_windows=3)

    low = e.evaluate(
        RiskInput(spoof_probability=0.99, quality_confidence_multiplier=0.1)
    )
    good1 = e.evaluate(RiskInput(spoof_probability=0.80))
    good2 = e.evaluate(RiskInput(spoof_probability=0.70))

    acc.add(low)
    acc.add(good1)
    acc.add(good2)

    assert len(acc.windows) == 2
    assert acc.aggregate() is not None


if __name__ == "__main__":
    for fn in [
        test_bonafide_good_quality,
        test_spoof_good_quality,
        test_bad_quality_moves_toward_neutral,
        test_poor_quality_does_not_create_spoof_evidence,
        test_prosody_is_secondary,
        test_temporal_accumulator_ignores_low_confidence,
    ]:
        fn()
    print("DYNAMIC RISK ENGINE TESTS PASS")
