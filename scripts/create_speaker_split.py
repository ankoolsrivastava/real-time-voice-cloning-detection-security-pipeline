import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT = Path("dataset_v1/metadata/speaker_inventory.csv")
OUTPUT = Path("dataset_v1/metadata/speaker_split.csv")

SEED = 42

df = pd.read_csv(INPUT)

splits = []

for language, group in df.groupby("language", sort=True):

    speakers = group["speaker_id"].tolist()

    train, temp = train_test_split(
        speakers,
        test_size=0.30,
        random_state=SEED,
        shuffle=True
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        shuffle=True
    )

    for speaker in train:
        splits.append({
            "speaker_id": speaker,
            "language": language,
            "split": "train"
        })

    for speaker in val:
        splits.append({
            "speaker_id": speaker,
            "language": language,
            "split": "validation"
        })

    for speaker in test:
        splits.append({
            "speaker_id": speaker,
            "language": language,
            "split": "test"
        })

result = pd.DataFrame(splits)

result.to_csv(OUTPUT, index=False)

print(f"Created: {OUTPUT}")
print()

print(result.groupby(["language", "split"]).size().unstack(fill_value=0))

