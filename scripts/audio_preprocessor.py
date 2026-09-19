from pathlib import Path

import librosa
import numpy as np
import torch


class VoiceGuardAudioPreprocessor:
    """
    VoiceGuard audio preprocessing pipeline.

    Input:
        Audio file path

    Output:
        Fixed-length 16 kHz mono waveform

    Processing:
        1. Load audio
        2. Convert to mono
        3. Resample to 16 kHz
        4. Convert to float32
        5. Normalize safely
        6. Pad or truncate to fixed duration
    """

    def __init__(
        self,
        target_sr=16000,
        max_duration_sec=10.0,
    ):
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec

        self.target_samples = int(
            target_sr * max_duration_sec
        )

    def load(self, audio_path):
        """
        Load audio as mono at target sample rate.
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sr = librosa.load(
            str(audio_path),
            sr=self.target_sr,
            mono=True,
        )

        waveform = waveform.astype(
            np.float32
        )

        return waveform, self.target_sr

    def normalize(self, waveform):
        """
        Peak normalization.

        Prevents recordings with different absolute
        amplitudes from dominating the model.
        """

        peak = np.max(
            np.abs(waveform)
        )

        if peak > 0:
            waveform = waveform / peak

        return waveform.astype(
            np.float32
        )

    def pad_or_truncate(self, waveform):
        """
        Convert waveform to exactly max_duration_sec.
        """

        current_length = len(waveform)

        if current_length > self.target_samples:

            waveform = waveform[
                :self.target_samples
            ]

        elif current_length < self.target_samples:

            padding = (
                self.target_samples
                - current_length
            )

            waveform = np.pad(
                waveform,
                (0, padding),
                mode="constant",
                constant_values=0,
            )

        return waveform.astype(
            np.float32
        )

    def process(self, audio_path):
        """
        Complete preprocessing pipeline.

        Returns:
            torch.FloatTensor
            Shape: [target_samples]
        """

        waveform, sr = self.load(
            audio_path
        )

        waveform = self.normalize(
            waveform
        )

        waveform = self.pad_or_truncate(
            waveform
        )

        return torch.from_numpy(
            waveform
        )
    def process_waveform(
        self,
        waveform,
        sample_rate,
    ):
        """
        Process an in-memory PCM waveform.

        Production input contract:
            float32 mono PCM
            sample_rate == target_sr
        """

        if sample_rate != self.target_sr:
            raise ValueError(
                f"Expected sample rate {self.target_sr} Hz, "
                f"got {sample_rate} Hz."
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        ).reshape(-1)

        if waveform.size == 0:
            raise ValueError(
                "Empty waveform."
            )

        if not np.isfinite(waveform).all():
            raise ValueError(
                "Waveform contains NaN or Inf."
            )

        waveform = self.normalize(
            waveform
        )

        waveform = self.pad_or_truncate(
            waveform
        )

        return torch.from_numpy(
            waveform
        )