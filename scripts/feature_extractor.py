import librosa
import numpy as np
import torch


class VoiceGuardFeatureExtractor:
    """
    VoiceGuard V1 Log-Mel Spectrogram Feature Extractor.

    Input:
        Fixed-length 16 kHz mono waveform.

    Output:
        Log-Mel spectrogram tensor.

    Fixed configuration:
        Sample rate       : 16000 Hz
        Window            : 25 ms
        Hop               : 10 ms
        Mel bins          : 64
        Frequency range   : 20–8000 Hz
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=64,
        fmin=20,
        fmax=8000,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def extract(self, waveform):
        """
        Convert waveform into Log-Mel spectrogram.

        Parameters
        ----------
        waveform : torch.Tensor or numpy.ndarray
            Shape: [samples]

        Returns
        -------
        torch.FloatTensor
            Shape: [n_mels, time_frames]
        """

        # --------------------------------------------------
        # Convert PyTorch tensor → NumPy
        # --------------------------------------------------

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()

        waveform = np.asarray(
            waveform,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Validate waveform
        # --------------------------------------------------

        if waveform.ndim != 1:
            raise ValueError(
                f"Expected 1-D waveform, got shape {waveform.shape}"
            )

        if len(waveform) == 0:
            raise ValueError(
                "Waveform is empty"
            )

        if not np.isfinite(waveform).all():
            raise ValueError(
                "Waveform contains NaN or Inf"
            )

        # --------------------------------------------------
        # Mel spectrogram
        # --------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=True,
            power=2.0,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # --------------------------------------------------
        # Convert power → decibel
        # --------------------------------------------------

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # --------------------------------------------------
        # Final numerical safety check
        # --------------------------------------------------

        if not np.isfinite(log_mel).all():
            raise ValueError(
                "Log-Mel spectrogram contains NaN or Inf"
            )

        return torch.from_numpy(
            log_mel.astype(np.float32)
        )

    def extract_with_channel(self, waveform):
        """
        Extract features and add CNN channel dimension.

        Output:
            [1, n_mels, time_frames]
        """

        features = self.extract(waveform)

        return features.unsqueeze(0)