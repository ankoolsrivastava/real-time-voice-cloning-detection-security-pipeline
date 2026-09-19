from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from config import (
    MASTER_MANIFEST,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    SAMPLE_RATE,
    MAX_DURATION_SEC,
    CHECKPOINT_DIR,
)

from model import VoiceGuardModel

from reproducibility import set_seed

from training_dataset import (
    VoiceGuardTrainingDataset,
    voiceguard_collate_fn,
)


# ==========================================================
# VOICEGUARD — TRAINING ENGINE
# ==========================================================

# Locked Master Dataset V1.
# This file is read-only and is the sole training source.
FINAL_MANIFEST = MASTER_MANIFEST


def validate_training_manifest(manifest_path):
    """
    Validate that the manifest is suitable for binary
    bonafide/spoof training.

    Training is intentionally refused unless both classes
    are present.
    """

    import pandas as pd

    if not manifest_path.exists():

        print()
        print("TRAINING NOT STARTED")
        print("-" * 70)
        print(
            "Final training manifest does not exist yet:"
        )
        print(
            f"  {manifest_path}"
        )
        print()
        print(
            "This is expected until REAL + SPOOF data "
            "are integrated."
        )
        return False

    df = pd.read_csv(manifest_path)

    if "label" not in df.columns:

        raise ValueError(
            "Training manifest is missing the 'label' column."
        )

    valid_quality_statuses = {
        "usable",
        "PASS",
        "STANDARDIZED",
    }

    usable = df[
        df["quality_status"].isin(
            valid_quality_statuses
        )
    ].copy()

    labels = set(
        usable["label"].dropna().astype(str)
    )

    print()
    print("Training manifest")
    print("-" * 70)
    print("Total rows     :", len(df))
    print("Usable rows    :", len(usable))
    print("Labels found   :", sorted(labels))

    has_bonafide = "bonafide" in labels
    has_spoof = "spoof" in labels

    if not has_bonafide or not has_spoof:

        print()
        print(
            "FINAL BINARY TRAINING NOT STARTED."
        )

        print(
            "Both bonafide and spoof classes are required."
        )

        print(
            f"Bonafide present : {has_bonafide}"
        )

        print(
            f"Spoof present    : {has_spoof}"
        )

        print()
        print(
            "No dummy labels or synthetic training data "
            "will be created."
        )

        return False

    return True


def create_dataloader(
    manifest_path,
    split,
    shuffle,
):
    """
    Create the actual VoiceGuard DataLoader.
    """

    dataset = VoiceGuardTrainingDataset(
        manifest_path=str(manifest_path),
        split=split,
        target_sr=SAMPLE_RATE,
        max_duration_sec=MAX_DURATION_SEC,
    )

    if len(dataset) == 0:

        raise ValueError(
            f"No usable samples found for split: {split}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=voiceguard_collate_fn,
    )

    return dataset, loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Run one complete training epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels, _ in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _ = model(
            features
        )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        loss = criterion(
            logits,
            labels,
        )

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        loss.backward()

        # --------------------------------------------------
        # Parameter update
        # --------------------------------------------------

        optimizer.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    if total == 0:

        raise RuntimeError(
            "Training loader produced zero samples."
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    """
    Run validation without gradient computation.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels, _ in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits, _ = model(
            features
        )

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    if total == 0:

        raise RuntimeError(
            "Validation loader produced zero samples."
        )

    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct / total
    )

    return (
        validation_loss,
        validation_accuracy,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    validation_accuracy,
):
    """
    Save a complete resumable training checkpoint.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy,

        "random_seed":
            RANDOM_SEED,
    }

    torch.save(
        checkpoint,
        path,
    )


def main():

    print("=" * 70)
    print("VOICEGUARD — TRAINING ENGINE")
    print("=" * 70)

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device :", device)

    if device.type == "cuda":

        print(
            "GPU    :",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------
    # Dataset gate
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET READINESS CHECK")
    print("=" * 70)

    ready = validate_training_manifest(
        FINAL_MANIFEST
    )

    if not ready:

        print()
        print("=" * 70)
        print("TRAINING ENGINE EXITED SAFELY")
        print("=" * 70)

        sys.exit(0)

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING DATA LOADERS")
    print("=" * 70)

    train_dataset, train_loader = (
        create_dataloader(
            FINAL_MANIFEST,
            "train",
            shuffle=True,
        )
    )

    validation_dataset, validation_loader = (
        create_dataloader(
            FINAL_MANIFEST,
            "validation",
            shuffle=False,
        )
    )

    print(
        "Train samples      :",
        len(train_dataset)
    )

    print(
        "Validation samples :",
        len(validation_dataset)
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = VoiceGuardModel().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model parameters :",
        parameter_count
    )

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        CHECKPOINT_DIR /
        "best_model.pt"
    )

    # ------------------------------------------------------
    # Training state
    # ------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    print()
    print("=" * 70)
    print("TRAINING START")
    print("=" * 70)

    # ------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"  Train loss      : {train_loss:.6f}"
        )

        print(
            f"  Train accuracy  : {train_accuracy:.4f}"
        )

        print(
            f"  Val loss        : {validation_loss:.6f}"
        )

        print(
            f"  Val accuracy    : {validation_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Best checkpoint
        # --------------------------------------------------

        if validation_loss < (
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                train_loss,
                validation_loss,
                validation_accuracy,
            )

            print(
                "  Checkpoint      : BEST SAVED"
            )

        else:

            epochs_without_improvement += 1

            print(
                "  No improvement  :",
                epochs_without_improvement,
                "/",
                EARLY_STOPPING_PATIENCE,
            )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best checkpoint :",
        best_checkpoint
    )

    print(
        "Best validation loss :",
        best_validation_loss
    )


if __name__ == "__main__":
    main()