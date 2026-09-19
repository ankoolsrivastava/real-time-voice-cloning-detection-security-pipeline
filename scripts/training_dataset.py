import torch
from torch.utils.data import Dataset

from dataset_loader import VoiceGuardDataset
from audio_preprocessor import VoiceGuardAudioPreprocessor
from feature_extractor import VoiceGuardFeatureExtractor


class VoiceGuardTrainingDataset(Dataset):
    """
    VoiceGuard training dataset pipeline.

    Pipeline:
        Manifest
            ↓
        Audio preprocessing
            ↓
        Log-Mel feature extraction
            ↓
        CNN-ready tensor

    Returns:
        features : [1, 64, time]
        label    : scalar torch.long
        metadata : dictionary
    """

    def __init__(
        self,
        manifest_path,
        split,
        target_sr=16000,
        max_duration_sec=10.0,
    ):
        self.base_dataset = VoiceGuardDataset(
            manifest_path=manifest_path,
            split=split,
            target_sr=target_sr,
            max_duration=max_duration_sec,
        )

        self.preprocessor = VoiceGuardAudioPreprocessor(
            target_sr=target_sr,
            max_duration_sec=max_duration_sec,
        )

        self.feature_extractor = VoiceGuardFeatureExtractor(
            sample_rate=target_sr,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Load one sample through the actual VoiceGuard pipeline.
        """

        # Get metadata and label from the manifest.
        # We intentionally do not use the waveform returned by
        # VoiceGuardDataset because the preprocessor performs the
        # authoritative fixed-length preprocessing.
        _, label, metadata = self.base_dataset[index]

        audio_path = metadata["audio_path"]

        # --------------------------------------------------
        # Actual preprocessing
        # --------------------------------------------------

        waveform = self.preprocessor.process(
            audio_path
        )

        # --------------------------------------------------
        # Actual Log-Mel extraction
        # --------------------------------------------------

        features = self.feature_extractor.extract_with_channel(
            waveform
        )

        # --------------------------------------------------
        # Numerical safety
        # --------------------------------------------------

        if not torch.isfinite(features).all():
            raise ValueError(
                f"Non-finite features detected for "
                f"{metadata['file_id']}"
            )

        return features, label, metadata


def voiceguard_collate_fn(batch):
    """
    Collate VoiceGuard samples into a training batch.

    Expected individual feature shape:
        [1, 64, time]

    Returned batch feature shape:
        [batch, 1, 64, time]
    """

    if not batch:
        raise ValueError("Received empty batch.")

    features = torch.stack(
        [item[0] for item in batch],
        dim=0,
    )

    labels = torch.stack(
        [item[1] for item in batch],
        dim=0,
    )

    metadata = [
        item[2]
        for item in batch
    ]

    return features, labels, metadata