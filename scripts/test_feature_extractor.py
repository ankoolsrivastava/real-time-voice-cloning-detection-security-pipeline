import torch

from audio_preprocessor import (
    VoiceGuardAudioPreprocessor
)

from feature_extractor import (
    VoiceGuardFeatureExtractor
)


AUDIO = (
    "dataset_v1/"
    "processed/real/hindi/"
    "REAL_000001.wav"
)


def main():

    print("=" * 70)
    print("VOICEGUARD — LOG-MEL FEATURE EXTRACTOR TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    processor = VoiceGuardAudioPreprocessor(
        target_sr=16000,
        max_duration_sec=10.0,
    )

    waveform = processor.process(
        AUDIO
    )

    print("\nWaveform")
    print("Shape :", waveform.shape)
    print("Dtype :", waveform.dtype)

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------

    extractor = VoiceGuardFeatureExtractor()

    features = extractor.extract(
        waveform
    )

    print("\nLog-Mel Features")
    print("Shape :", features.shape)
    print("Dtype :", features.dtype)

    # --------------------------------------------------
    # CNN format
    # --------------------------------------------------

    cnn_features = extractor.extract_with_channel(
        waveform
    )

    print("\nCNN Input")
    print("Shape :", cnn_features.shape)

    # --------------------------------------------------
    # Numerical checks
    # --------------------------------------------------

    print("\nNumerical checks")

    print(
        "Contains NaN :",
        torch.isnan(features).any().item()
    )

    print(
        "Contains Inf :",
        torch.isinf(features).any().item()
    )

    print(
        "Minimum      :",
        features.min().item()
    )

    print(
        "Maximum      :",
        features.max().item()
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    if features.ndim != 2:
        raise RuntimeError(
            "Feature tensor must be 2-D"
        )

    if features.shape[0] != 64:
        raise RuntimeError(
            f"Expected 64 Mel bins, got {features.shape[0]}"
        )

    if features.dtype != torch.float32:
        raise RuntimeError(
            "Feature tensor must be float32"
        )

    if torch.isnan(features).any():
        raise RuntimeError(
            "NaN detected in features"
        )

    if torch.isinf(features).any():
        raise RuntimeError(
            "Inf detected in features"
        )

    if cnn_features.shape[0] != 1:
        raise RuntimeError(
            "CNN channel dimension incorrect"
        )

    print(
        "\nPASS — Log-Mel feature extraction is valid"
    )


if __name__ == "__main__":
    main()