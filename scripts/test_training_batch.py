import torch
from torch.utils.data import DataLoader

from training_dataset import (
    VoiceGuardTrainingDataset,
    voiceguard_collate_fn,
)

from model import VoiceGuardModel


MANIFEST = (
    "dataset_v1/metadata/"
    "real_master_manifest.csv"
)


def main():

    print("=" * 70)
    print("VOICEGUARD — COMPLETE TRAINING BATCH TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = VoiceGuardTrainingDataset(
        manifest_path=MANIFEST,
        split="train",
    )

    print(
        "\nTraining samples:",
        len(dataset)
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=voiceguard_collate_fn,
    )

    # --------------------------------------------------
    # Get one real batch
    # --------------------------------------------------

    features, labels, metadata = next(
        iter(loader)
    )

    print("\nBatch")
    print("Features :", features.shape)
    print("Labels   :", labels.shape)

    print(
        "Label values:",
        labels.tolist()
    )

    print(
        "Feature dtype:",
        features.dtype
    )

    # --------------------------------------------------
    # Numerical checks
    # --------------------------------------------------

    if not torch.isfinite(features).all():

        raise RuntimeError(
            "Batch contains NaN or Inf"
        )

    # --------------------------------------------------
    # Move to GPU
    # --------------------------------------------------

    features = features.to(
        device,
        non_blocking=True
    )

    labels = labels.to(
        device,
        non_blocking=True
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = VoiceGuardModel().to(
        device
    )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    logits, attention = model(
        features
    )

    print("\nModel output")
    print("Logits    :", logits.shape)
    print("Attention :", attention.shape)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels
    )

    print(
        "Loss      :",
        loss.item()
    )

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    model.zero_grad()

    loss.backward()

    print(
        "Backward  : PASS"
    )

    # --------------------------------------------------
    # GPU memory
    # --------------------------------------------------

    if device.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**2
        )

        print(
            f"\nGPU allocated : {allocated:.2f} MB"
        )

        print(
            f"GPU reserved  : {reserved:.2f} MB"
        )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TRAINING PIPELINE TEST")
    print("=" * 70)

    print(
        "✅ REAL AUDIO → FEATURES → BATCH → MODEL → LOSS → BACKWARD"
    )

    print(
        "✅ COMPLETE TRAINING PATH IS OPERATIONAL"
    )


if __name__ == "__main__":
    main()