import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.audio.types import AudioChunk
from app.audio.stream_processor import AudioStreamProcessor


DEVICE = 1
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
TOTAL_SECONDS = 10

OUTPUT_PATH = Path(r"D:\VoiceGaurd\temp\live_demo.wav")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 76)
    print("                 VOICEGUARD LIVE DEMO")
    print("=" * 76)
    print(f"Microphone : {DEVICE}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Window     : {TOTAL_SECONDS} seconds")
    print()
    print("Speak normally for the entire 10 seconds.")
    print()

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    processor = AudioStreamProcessor()

    recorded_chunks = []
    results = []

    print()
    print("🎤 LISTENING...")
    print()

    wall_start = time.monotonic()

    for seq in range(TOTAL_SECONDS):

        media_start = seq * CHUNK_SECONDS
        media_end = media_start + CHUNK_SECONDS

        audio = sd.rec(
            CHUNK_SAMPLES,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=DEVICE,
        )

        sd.wait()

        # Arrival timestamp MUST be taken after capture completes.
        received_at = time.monotonic()

        audio = np.asarray(
            audio,
            dtype=np.float32
        ).reshape(-1)

        recorded_chunks.append(audio.copy())

        chunk = AudioChunk(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            sequence_number=seq,
            timestamp_start=media_start,
            timestamp_end=media_end,
            received_at=received_at,
        )

        result = processor.add_chunk(chunk)

        print(
            f"  Chunk {seq + 1:02d}/{TOTAL_SECONDS} | "
            f"{len(audio):5d} samples | "
            f"buffered={processor.window_manager.buffered_samples:6d}"
        )

        if result is not None:
            results.append(result)

    elapsed = time.monotonic() - wall_start

    raw_audio = np.concatenate(recorded_chunks).astype(np.float32)

    sf.write(
        OUTPUT_PATH,
        raw_audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    print()
    print("=" * 76)
    print("                         RESULT")
    print("=" * 76)

    if not results:
        print("ERROR: No complete 10-second window was produced.")
        return

    r = results[-1].result

    print()
    print(f"  Prediction        : {r.prediction_label}")
    print(f"  Bonafide          : {r.bonafide_probability:.4f}")
    print(f"  Spoof             : {r.spoof_probability:.4f}")
    print()
    print(f"  Audio Quality     : {r.quality.quality_score:.4f}")
    print(f"  Quality Status    : {r.quality.status}")
    print(f"  Confidence        : {r.quality.confidence_multiplier:.4f}")
    print()
    print(f"  Risk Score        : {r.risk.risk_score:.2f} / 100")
    print(f"  Risk Level        : {r.risk.risk_level}")
    print(f"  Risk Status       : {r.risk.status}")
    print(f"  Evidence          : {r.risk.evidence_confidence:.4f}")
    print()

    if r.risk.risk_level == "LOW":
        print("  ACTION            : ALLOW")
    elif r.risk.risk_level == "MEDIUM":
        print("  ACTION            : VERIFY")
    elif r.risk.risk_level == "HIGH":
        print("  ACTION            : SECONDARY VERIFICATION")
    else:
        print("  ACTION            : BLOCK / ESCALATE")

    print()
    print("=" * 76)
    print("                        TELEMETRY")
    print("=" * 76)

    telemetry = processor.telemetry()

    print(
        f"  Chunks received   : "
        f"{telemetry['buffer']['total_chunks_received']}"
    )
    print(
        f"  Missing chunks    : "
        f"{telemetry['buffer']['total_missing_chunks']}"
    )
    print(
        f"  Duplicate chunks  : "
        f"{telemetry['buffer']['total_duplicate_chunks']}"
    )
    print(
        f"  Out-of-order      : "
        f"{telemetry['buffer']['total_out_of_order_chunks']}"
    )
    print(
        f"  Packet loss       : "
        f"{telemetry['buffer']['packet_loss_ratio']:.4f}"
    )

    print()
    print(f"  Model             : {r.model_version}")
    print(f"  Device            : {r.device}")
    print(f"  Elapsed           : {elapsed:.3f} sec")
    print(f"  Saved recording   : {OUTPUT_PATH}")

    print("=" * 76)


if __name__ == "__main__":
    main()
