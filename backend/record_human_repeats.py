import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

DEVICE = 2
SR = 16000
SECONDS = 10
SAMPLES = SR * SECONDS

out_dir = Path(r"D:\VoiceGaurd\temp\human_repeats")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("VOICEGUARD — 3 HUMAN BASELINE RECORDINGS")
print("=" * 70)

for attempt in range(1, 4):

    print()
    print(f"ATTEMPT {attempt}/3")
    print("Speak naturally in Hindi/Marathi for 10 seconds.")

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    print("🎤 RECORDING...")

    audio = sd.rec(
        SAMPLES,
        samplerate=SR,
        channels=1,
        dtype="float32",
        device=DEVICE,
    )
    sd.wait()

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    path = out_dir / f"human_{attempt}.wav"

    sf.write(
        path,
        audio,
        SR,
        subtype="PCM_16",
    )

    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    peak = np.max(np.abs(audio))

    print(f"Saved: {path}")
    print(f"RMS:   {rms:.6f}")
    print(f"Peak:  {peak:.6f}")

print()
print("DONE — 3 recordings created.")
