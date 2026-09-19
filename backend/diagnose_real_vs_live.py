import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"D:\VOICEGUARD_ML_HANDOFF_V2\runtime")
sys.path.insert(0, r"D:\VOICEGUARD_ML_HANDOFF_V2\runtime\scripts")

from scripts.inference import VoiceGuardInference

manifest = r"D:\VoiceGaurd\master_v1\metadata\master_dataset_v1.csv"

df = pd.read_csv(manifest)

x = df[
    (df["label"] == "bonafide")
    & (df["split"].isin(["train", "validation"]))
    & (df["quality_status"].str.lower() == "usable")
].drop_duplicates("speaker_id")

hindi = x[x["language_name"].eq("Hindi")].head(6)
marathi = x[x["language_name"].eq("Marathi")].head(6)
samples = pd.concat([hindi, marathi])

print()
print("APPROVED REAL vs FROZEN V2")
print("=" * 100)
print(
    f"{'FILE':<15}"
    f"{'LANG':<10}"
    f"{'SPLIT':<12}"
    f"{'PRED':<12}"
    f"{'SPOOF_PROB':>14}"
    f"{'BONAFIDE_PROB':>16}"
)
print("-" * 100)

inference = VoiceGuardInference(
    checkpoint_path=r"D:\VOICEGUARD_ML_HANDOFF_V2\models\best_model.pt"
)

results = []

for _, row in samples.iterrows():
    result = inference.predict(row["processed_audio_path"])

    results.append({
        "file": row["file_id"],
        "language": row["language_name"],
        "split": row["split"],
        "prediction": result["prediction"],
        "spoof_probability": result["spoof_probability"],
        "bonafide_probability": result["bonafide_probability"],
    })

    print(
        f"{row['file_id']:<15}"
        f"{row['language_name']:<10}"
        f"{row['split']:<12}"
        f"{result['prediction']:<12}"
        f"{result['spoof_probability']:>14.6f}"
        f"{result['bonafide_probability']:>16.6f}"
    )

spoof_probs = np.array(
    [r["spoof_probability"] for r in results],
    dtype=float
)

print("-" * 100)
print(f"MEAN spoof probability:   {spoof_probs.mean():.6f}")
print(f"MEDIAN spoof probability: {np.median(spoof_probs):.6f}")
print(f"MIN spoof probability:    {spoof_probs.min():.6f}")
print(f"MAX spoof probability:    {spoof_probs.max():.6f}")

bonafide_count = sum(
    r["prediction"].lower() == "bonafide"
    for r in results
)

spoof_count = sum(
    r["prediction"].lower() == "spoof"
    for r in results
)

print(f"BONAFIDE predictions:     {bonafide_count}/12")
print(f"SPOOF predictions:        {spoof_count}/12")

print()
print("LIVE MIC REFERENCE")
print("=" * 100)

live = inference.predict(
    r"D:\VoiceGaurd\temp\live_debug.wav"
)

print(f"Spoof probability:        {live['spoof_probability']:.6f}")
print(f"Bonafide probability:     {live['bonafide_probability']:.6f}")
print(f"Prediction:               {live['prediction']}")

print("=" * 100)
