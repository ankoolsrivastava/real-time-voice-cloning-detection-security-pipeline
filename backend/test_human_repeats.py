import sys
from pathlib import Path

sys.path.insert(0, r"D:\VOICEGUARD_ML_HANDOFF_V2\runtime")
sys.path.insert(0, r"D:\VOICEGUARD_ML_HANDOFF_V2\runtime\scripts")

from scripts.inference import VoiceGuardInference

files = [
    r"D:\VoiceGaurd\temp\human_repeats\human_1.wav",
    r"D:\VoiceGaurd\temp\human_repeats\human_2.wav",
    r"D:\VoiceGaurd\temp\human_repeats\human_3.wav",
]

inference = VoiceGuardInference(
    checkpoint_path=r"D:\VOICEGUARD_ML_HANDOFF_V2\models\best_model.pt"
)

print()
print("HUMAN LIVE-MIC BASELINE")
print("=" * 80)

for path in files:
    result = inference.predict(path)

    print(
        f"{Path(path).name:<15} "
        f"prediction={result['prediction']:<8} "
        f"bonafide={result['bonafide_probability']:.6f} "
        f"spoof={result['spoof_probability']:.6f}"
    )

print("=" * 80)
