import numpy as np
from voiceguard_audio_quality_interface_v1 import assess_audio_quality


def main():
    sr = 16000

    good = 0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    r = assess_audio_quality(good, sr)
    assert 0 <= r.quality_score <= 1
    assert r.status in {"GOOD", "DEGRADED", "POOR", "INVALID"}

    empty = assess_audio_quality(np.array([], dtype=np.float32), sr)
    assert empty.status == "INVALID"
    assert empty.confidence_multiplier == 0

    lossy = assess_audio_quality(
        good, sr, packet_loss_ratio=0.20, jitter_ms=80, codec_degradation_score=0.8
    )
    assert lossy.quality_score < r.quality_score

    clipped = np.ones(sr, dtype=np.float32)
    cr = assess_audio_quality(clipped, sr)
    assert cr.clipping_ratio > 0.9
    assert cr.quality_score < r.quality_score

    print("AUDIO QUALITY INTERFACE TESTS PASS")


if __name__ == "__main__":
    main()
