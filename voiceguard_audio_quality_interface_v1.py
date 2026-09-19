from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import math
import numpy as np


@dataclass
class AudioQualityResult:
    quality_score: float
    confidence_multiplier: float
    status: str
    rms_db: float
    peak: float
    clipping_ratio: float
    duration_sec: float
    active_ratio: float
    packet_loss_ratio: Optional[float] = None
    jitter_ms: Optional[float] = None
    codec_degradation_score: Optional[float] = None
    notes: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def assess_audio_quality(
    waveform,
    sample_rate: int = 16000,
    packet_loss_ratio: float | None = None,
    jitter_ms: float | None = None,
    codec_degradation_score: float | None = None,
) -> AudioQualityResult:
    """
    Deterministic baseline quality assessor.

    Important:
    - This module does NOT label audio as spoof.
    - Quality degradation only lowers confidence.
    - Network indicators are optional because offline/local inference may not have them.
    """
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration = len(y) / sample_rate if sample_rate else 0.0

    if len(y) == 0 or sample_rate <= 0:
        return AudioQualityResult(
            quality_score=0.0,
            confidence_multiplier=0.0,
            status="INVALID",
            rms_db=-120.0,
            peak=0.0,
            clipping_ratio=0.0,
            duration_sec=0.0,
            active_ratio=0.0,
            packet_loss_ratio=packet_loss_ratio,
            jitter_ms=jitter_ms,
            codec_degradation_score=codec_degradation_score,
            notes=["Empty or invalid waveform."]
        )

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y * y) + 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-6))

    # Treat near-zero samples as inactive for a conservative activity proxy.
    active_ratio = float(np.mean(np.abs(y) > max(1e-4, peak * 0.02)))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.995))

    # Baseline component scores. These are engineering heuristics, not trained probabilities.
    duration_score = _clip01(duration / 1.0)
    activity_score = _clip01(active_ratio / 0.20)
    level_score = _clip01((rms_db + 50.0) / 35.0)
    clipping_score = _clip01(1.0 - clipping_ratio / 0.01)

    scores = [duration_score, activity_score, level_score, clipping_score]

    if packet_loss_ratio is not None:
        scores.append(_clip01(1.0 - packet_loss_ratio))
    if jitter_ms is not None:
        # Conservative normalization: 0 ms is best; >=100 ms is fully degraded.
        scores.append(_clip01(1.0 - jitter_ms / 100.0))
    if codec_degradation_score is not None:
        scores.append(_clip01(1.0 - codec_degradation_score))

    quality = float(np.mean(scores))

    if quality >= 0.75:
        status = "GOOD"
    elif quality >= 0.50:
        status = "DEGRADED"
    elif quality > 0.0:
        status = "POOR"
    else:
        status = "INVALID"

    notes = []
    if clipping_ratio > 0.01:
        notes.append("Significant clipping/saturation detected.")
    if active_ratio < 0.05:
        notes.append("Low speech/activity proxy; window may be mostly silence.")
    if packet_loss_ratio is not None and packet_loss_ratio > 0.05:
        notes.append("Packet loss is reducing evidence confidence.")
    if jitter_ms is not None and jitter_ms > 50:
        notes.append("High jitter is reducing evidence confidence.")
    if codec_degradation_score is not None and codec_degradation_score > 0.5:
        notes.append("Codec/compression degradation is reducing evidence confidence.")

    return AudioQualityResult(
        quality_score=quality,
        confidence_multiplier=quality,
        status=status,
        rms_db=rms_db,
        peak=peak,
        clipping_ratio=clipping_ratio,
        duration_sec=duration,
        active_ratio=active_ratio,
        packet_loss_ratio=packet_loss_ratio,
        jitter_ms=jitter_ms,
        codec_degradation_score=codec_degradation_score,
        notes=notes,
    )
