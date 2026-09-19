from pathlib import Path
import sys
import torch
import numpy as np

from dataset_loader import VoiceGuardDataset


MANIFEST = "dataset_v1/metadata/real_master_manifest.csv"


def main():

    print("=" * 70)
    print("VOICEGUARD — FULL REAL DATASET LOADER TEST")
    print("=" * 70)

    dataset = VoiceGuardDataset(MANIFEST)

    print("\nDataset size:", len(dataset))

    errors = []
    nan_count = 0
    inf_count = 0

    for i in range(len(dataset)):

        try:
            waveform, label, metadata = dataset[i]

            # --------------------------------------------
            # Tensor checks
            # --------------------------------------------

            if not isinstance(waveform, torch.Tensor):
                errors.append(
                    (i, "waveform is not a tensor")
                )
                continue

            if waveform.dtype != torch.float32:
                errors.append(
                    (
                        i,
                        f"wrong dtype: {waveform.dtype}"
                    )
                )

            # --------------------------------------------
            # NaN / Inf checks
            # --------------------------------------------

            if torch.isnan(waveform).any():
                nan_count += 1
                errors.append(
                    (i, "NaN detected")
                )

            if torch.isinf(waveform).any():
                inf_count += 1
                errors.append(
                    (i, "Inf detected")
                )

            # --------------------------------------------
            # Label check
            # --------------------------------------------

            if label.item() not in [0, 1]:
                errors.append(
                    (
                        i,
                        f"invalid label: {label.item()}"
                    )
                )

            # --------------------------------------------
            # Audio check
            # --------------------------------------------

            if waveform.numel() == 0:
                errors.append(
                    (i, "empty waveform")
                )

        except Exception as exc:

            errors.append(
                (i, str(exc))
            )

        # Progress

        if (i + 1) % 250 == 0:
            print(
                f"Checked: {i + 1}/{len(dataset)}"
            )

    # ----------------------------------------------------
    # Final report
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL LOADER TEST")
    print("=" * 70)

    print("Total samples :", len(dataset))
    print("Errors        :", len(errors))
    print("NaN samples   :", nan_count)
    print("Inf samples   :", inf_count)

    if errors:

        print("\nFirst errors:")

        for error in errors[:20]:
            print(" ", error)

        print(
            "\n❌ DATASET LOADER TEST FAILED"
        )

        sys.exit(1)

    print(
        "\n✅ ALL REAL AUDIO SAMPLES LOADED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()