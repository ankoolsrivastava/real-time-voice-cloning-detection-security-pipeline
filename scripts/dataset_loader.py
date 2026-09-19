from pathlib import Path
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class VoiceGuardDataset(Dataset):
    """
    VoiceGuard Dataset V1 loader.

    Loads audio using the master manifest and returns:
        audio waveform
        label
        metadata
    """

    LABEL_MAP = {
        "bonafide": 0,
        "spoof": 1,
    }

    def __init__(
        self,
        manifest_path,
        split=None,
        target_sr=16000,
        max_duration=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.target_sr = target_sr
        self.max_duration = max_duration

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.df = pd.read_csv(self.manifest_path)

        # --------------------------------------------------
        # Required master columns
        # --------------------------------------------------

        required_columns = [
            "file_id",
            "speaker_id",
            "language",
            "language_name",
            "split",
            "label",
            "processed_audio_path",
            "duration_sec",
            "sample_rate",
            "channels",
            "quality_status",
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}"
            )

        # --------------------------------------------------
        # Keep Master V1 approved samples
        # --------------------------------------------------

        valid_quality_statuses = {
            "usable",
            "PASS",
            "STANDARDIZED",
        }

        self.df = self.df[
            self.df["quality_status"].isin(
                valid_quality_statuses
            )
        ].copy()

        # --------------------------------------------------
        # Optional split filter
        # --------------------------------------------------

        if split is not None:

            valid_splits = {
                "train",
                "validation",
                "test",
            }

            if split not in valid_splits:
                raise ValueError(
                    f"Invalid split: {split}"
                )

            self.df = self.df[
                self.df["split"] == split
            ].copy()

        # --------------------------------------------------
        # Validate labels
        # --------------------------------------------------

        unknown_labels = set(
            self.df["label"].dropna()
        ) - set(self.LABEL_MAP)

        if unknown_labels:
            raise ValueError(
                f"Unknown labels: {unknown_labels}"
            )

        # --------------------------------------------------
        # Reset index
        # --------------------------------------------------

        self.df.reset_index(
            drop=True,
            inplace=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        audio_path = Path(
            row["processed_audio_path"]
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio not found: {audio_path}"
            )

        # --------------------------------------------------
        # Load audio
        # --------------------------------------------------

        waveform, sr = librosa.load(
            str(audio_path),
            sr=self.target_sr,
            mono=True,
        )

        # --------------------------------------------------
        # Optional duration limit
        # --------------------------------------------------

        if self.max_duration is not None:

            max_samples = int(
                self.target_sr *
                self.max_duration
            )

            if len(waveform) > max_samples:

                waveform = waveform[:max_samples]

        # --------------------------------------------------
        # Convert to float32
        # --------------------------------------------------

        waveform = waveform.astype(
            np.float32
        )

        # --------------------------------------------------
        # Convert to PyTorch tensor
        # --------------------------------------------------

        waveform = torch.from_numpy(
            waveform
        )

        label = torch.tensor(
            self.LABEL_MAP[row["label"]],
            dtype=torch.long
        )

        metadata = {
            "file_id": row["file_id"],
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "language_name": row["language_name"],
            "split": row["split"],
            "label": row["label"],
            "audio_path": str(audio_path),
            "duration_sec": float(
                row["duration_sec"]
            ),
        }

        return waveform, label, metadata