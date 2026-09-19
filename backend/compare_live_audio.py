import soundfile as sf
import numpy as np
from pathlib import Path

files = [
    r"D:\VoiceGaurd\temp\live_debug.wav",
    r"D:\VoiceGaurd\temp\live_demo.wav",
]

print("LIVE AUDIO DIAGNOSTIC")
print("=" * 90)

for path in files:
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float64)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    print()
    print(Path(path).name)
    print(f"  sample rate : {sr}")
    print(f"  samples     : {len(audio)}")
    print(f"  duration    : {len(audio) / sr:.3f} sec")
    print(f"  RMS         : {rms:.6f}")
    print(f"  RMS dB      : {20*np.log10(max(rms, 1e-12)):.3f}")
    print(f"  peak        : {peak:.6f}")
    print(f"  mean        : {np.mean(audio):.6f}")
    print(f"  std         : {np.std(audio):.6f}")
    print(f"  clipping    : {np.mean(np.abs(audio) >= 0.999):.6f}")
