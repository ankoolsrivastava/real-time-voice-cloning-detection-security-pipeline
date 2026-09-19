import torch

from audio_preprocessor import (
    VoiceGuardAudioPreprocessor
)


AUDIO = (
    "dataset_v1/"
    "processed/real/hindi/"
    "REAL_000001.wav"
)


def main():

    processor = VoiceGuardAudioPreprocessor(
        target_sr=16000,
        max_duration_sec=10.0,
    )

    waveform = processor.process(
        AUDIO
    )

    print("=" * 70)
    print("VOICEGUARD — AUDIO PREPROCESSOR TEST")
    print("=" * 70)

    print("Tensor shape :", waveform.shape)
    print("Dtype        :", waveform.dtype)
    print("Expected     :", torch.Size([160000]))

    print("Min          :", waveform.min().item())
    print("Max          :", waveform.max().item())

    print(
        "Contains NaN :",
        torch.isnan(waveform).any().item()
    )

    print(
        "Contains Inf :",
        torch.isinf(waveform).any().item()
    )

    if waveform.shape != torch.Size([160000]):
        raise RuntimeError(
            "Incorrect waveform length"
        )

    if waveform.dtype != torch.float32:
        raise RuntimeError(
            "Incorrect waveform dtype"
        )

    if torch.isnan(waveform).any():
        raise RuntimeError(
            "NaN detected"
        )

    if torch.isinf(waveform).any():
        raise RuntimeError(
            "Inf detected"
        )

    print()
    print(
        "PASS — preprocessing output is valid"
    )


if __name__ == "__main__":
    main()