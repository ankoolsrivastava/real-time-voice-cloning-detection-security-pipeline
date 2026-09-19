import pandas as pd
from pathlib import Path
import subprocess
import sys

MANIFEST = Path("dataset_v1/metadata/real_audio_manifest.csv")

manifest = pd.read_csv(MANIFEST)
manifest = manifest[manifest["quality_status"] == "usable"].copy()

success = 0
failed = []

for i, row in manifest.iterrows():

    language = row["language"]
    source = Path(row["source_path"])

    if language == "HI":
        out_dir = Path("dataset_v1/processed/real/hindi")
    elif language == "MR":
        out_dir = Path("dataset_v1/processed/real/marathi")
    else:
        continue

    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / f"{row['file_id']}.wav"

    if output.exists():
        success += 1
        continue

    command = [
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(output)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        failed.append({
            "file_id": row["file_id"],
            "source": str(source),
            "error": result.stderr[-500:]
        })
        continue

    success += 1

    if success % 100 == 0:
        print(f"Converted: {success}/{len(manifest)}")

print()
print("=== CONVERSION COMPLETE ===")
print("Successful:", success)
print("Failed:", len(failed))

if failed:
    failure_df = pd.DataFrame(failed)
    failure_df.to_csv(
        "dataset_v1/metadata/real_conversion_failures.csv",
        index=False
    )
    print("Failure report:")
    print("dataset_v1/metadata/real_conversion_failures.csv")

