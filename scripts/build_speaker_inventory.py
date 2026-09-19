import pandas as pd
from pathlib import Path

datasets = {
    "HI": Path("dataset_v1/raw/real/hindi/validated.tsv"),
    "MR": Path("dataset_v1/raw/real/marathi/validated.tsv")
}

out_dir = Path("dataset_v1/metadata")
out_dir.mkdir(parents=True, exist_ok=True)

all_rows = []

def mode_or_unknown(series):
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return "UNKNOWN"
    return s.mode().iloc[0]

for lang, path in datasets.items():
    df = pd.read_csv(path, sep="\t")

    # Internal speaker ID: language prefix prevents collisions between datasets.
    df["speaker_id"] = lang + "_" + df["client_id"].astype(str)

    grouped = []

    for speaker_id, g in df.groupby("speaker_id", sort=False):
        grouped.append({
            "speaker_id": speaker_id,
            "language": lang,
            "clip_count": len(g),
            "gender": mode_or_unknown(g["gender"]),
            "age": mode_or_unknown(g["age"]),
            "accent": mode_or_unknown(g["accents"]),
        })

    speaker_df = pd.DataFrame(grouped)
    all_rows.append(speaker_df)

inventory = pd.concat(all_rows, ignore_index=True)

inventory = inventory.sort_values(
    ["language", "clip_count", "speaker_id"],
    ascending=[True, False, True]
)

output = out_dir / "speaker_inventory.csv"
inventory.to_csv(output, index=False)

print(f"Created: {output}")
print(f"Total speakers: {len(inventory)}")
print()

for lang in inventory["language"].unique():
    s = inventory[inventory["language"] == lang]
    print(f"=== {lang} ===")
    print(f"Speakers: {len(s)}")
    print(f"Total clips: {s['clip_count'].sum()}")
    print(f"Median clips/speaker: {s['clip_count'].median()}")
    print(f"Mean clips/speaker: {s['clip_count'].mean():.2f}")
    print()
    print(s["gender"].value_counts(dropna=False).to_string())
    print()

