from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from config import (
    MASTER_MANIFEST,
    BATCH_SIZE,
    SAMPLE_RATE,
    MAX_DURATION_SEC,
    CHECKPOINT_DIR,
)

from model import VoiceGuardModel

from training_dataset import (
    VoiceGuardTrainingDataset,
    voiceguard_collate_fn,
)


# ==========================================================
# VOICEGUARD — EVALUATION PIPELINE
# ==========================================================

FINAL_MANIFEST = MASTER_MANIFEST

BEST_CHECKPOINT = (
    Path(CHECKPOINT_DIR) /
    "best_model.pt"
)


# ==========================================================
# DATA READINESS
# ==========================================================

def validate_evaluation_manifest(manifest_path):
    """
    Verify that the final manifest contains both classes.

    Evaluation is refused unless both bonafide and spoof
    samples are present.
    """

    if not manifest_path.exists():

        print()
        print("EVALUATION NOT STARTED")
        print("-" * 70)
        print(
            f"Final manifest not found:\n{manifest_path}"
        )
        print()
        print(
            "Waiting for REAL + SPOOF integration."
        )

        return False

    df = pd.read_csv(manifest_path)

    required_columns = {
        "label",
        "quality_status",
        "split",
    }

    missing = (
        required_columns -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Manifest missing required columns: "
            f"{sorted(missing)}"
        )

    valid_quality_statuses = {
        "usable",
        "PASS",
        "STANDARDIZED",
    }

    usable = df[
        df["quality_status"].isin(valid_quality_statuses)
    ].copy()

    labels = set(
        usable["label"]
        .dropna()
        .astype(str)
    )

    print()
    print("Evaluation manifest")
    print("-" * 70)
    print("Total rows :", len(df))
    print("Usable rows:", len(usable))
    print("Labels     :", sorted(labels))

    if (
        "bonafide" not in labels
        or
        "spoof" not in labels
    ):

        print()
        print(
            "EVALUATION NOT STARTED"
        )
        print(
            "Both bonafide and spoof classes "
            "are required."
        )
        print(
            "Bonafide present:",
            "bonafide" in labels
        )
        print(
            "Spoof present:",
            "spoof" in labels
        )

        return False

    test_rows = usable[
        usable["split"] == "test"
    ]

    if len(test_rows) == 0:

        raise ValueError(
            "Final manifest contains no usable test samples."
        )

    test_labels = set(
        test_rows["label"]
        .astype(str)
    )

    if (
        "bonafide" not in test_labels
        or
        "spoof" not in test_labels
    ):

        raise ValueError(
            "Test split must contain both "
            "bonafide and spoof samples."
        )

    print(
        "Test rows  :",
        len(test_rows)
    )

    return True


# ==========================================================
# CHECKPOINT
# ==========================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load the best trained VoiceGuard checkpoint.
    """

    if not checkpoint_path.exists():

        print()
        print("EVALUATION NOT STARTED")
        print("-" * 70)
        print(
            f"Checkpoint not found:\n{checkpoint_path}"
        )
        print()
        print(
            "A trained model is required before "
            "evaluation can begin."
        )

        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise ValueError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        "Checkpoint loaded successfully."
    )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "validation_loss" in checkpoint:

        print(
            "Validation loss:",
            checkpoint["validation_loss"]
        )

    if "validation_accuracy" in checkpoint:

        print(
            "Validation accuracy:",
            checkpoint["validation_accuracy"]
        )

    return checkpoint


# ==========================================================
# TEST DATALOADER
# ==========================================================

def create_test_loader(
    manifest_path,
):
    """
    Create the test-only VoiceGuard dataset.
    """

    dataset = VoiceGuardTrainingDataset(
        manifest_path=str(manifest_path),
        split="test",
        target_sr=SAMPLE_RATE,
        max_duration_sec=MAX_DURATION_SEC,
    )

    if len(dataset) == 0:

        raise ValueError(
            "Test dataset is empty."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=voiceguard_collate_fn,
    )

    return dataset, loader


# ==========================================================
# EER
# ==========================================================

def calculate_eer(
    labels,
    spoof_scores,
):
    """
    Calculate Equal Error Rate.

    labels:
        0 = bonafide
        1 = spoof

    spoof_scores:
        probability of spoof
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        spoof_scores,
        dtype=np.float64,
    )

    thresholds = np.unique(
        scores
    )

    thresholds = np.concatenate(
        [
            [0.0],
            thresholds,
            [1.0],
        ]
    )

    far_values = []
    frr_values = []

    for threshold in thresholds:

        predicted_spoof = (
            scores >= threshold
        )

        false_accepts = np.sum(
            (labels == 0)
            &
            predicted_spoof
        )

        bonafide_total = np.sum(
            labels == 0
        )

        false_rejects = np.sum(
            (labels == 1)
            &
            (~predicted_spoof)
        )

        spoof_total = np.sum(
            labels == 1
        )

        far = (
            false_accepts /
            bonafide_total
            if bonafide_total > 0
            else 0.0
        )

        frr = (
            false_rejects /
            spoof_total
            if spoof_total > 0
            else 0.0
        )

        far_values.append(far)
        frr_values.append(frr)

    far_values = np.asarray(
        far_values
    )

    frr_values = np.asarray(
        frr_values
    )

    difference = np.abs(
        far_values - frr_values
    )

    index = np.argmin(
        difference
    )

    eer = (
        far_values[index]
        +
        frr_values[index]
    ) / 2.0

    return float(eer)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
):
    """
    Run inference over the untouched test split.
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_spoof_scores = []

    for features, labels, _ in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        logits, _ = model(
            features
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        spoof_scores = (
            probabilities[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_spoof_scores.extend(
            spoof_scores
        )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_spoof_scores),
    )


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(
    labels,
    predictions,
    spoof_scores,
):
    """
    Calculate final binary anti-spoofing metrics.
    """

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        spoof_scores,
    )

    eer = calculate_eer(
        labels,
        spoof_scores,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "confusion_matrix": matrix,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "VOICEGUARD — EVALUATION PIPELINE"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------
    # Manifest gate
    # ------------------------------------------------------

    if not validate_evaluation_manifest(
        FINAL_MANIFEST
    ):

        print()
        print("=" * 70)

        return

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = VoiceGuardModel()

    model = model.to(
        device
    )

    # ------------------------------------------------------
    # Checkpoint gate
    # ------------------------------------------------------

    checkpoint = load_checkpoint(
        model,
        BEST_CHECKPOINT,
        device,
    )

    if checkpoint is None:

        print()
        print("=" * 70)

        return

    # ------------------------------------------------------
    # Test dataset
    # ------------------------------------------------------

    test_dataset, test_loader = (
        create_test_loader(
            FINAL_MANIFEST
        )
    )

    print()
    print(
        "Test samples:",
        len(test_dataset)
    )

    # ------------------------------------------------------
    # Inference
    # ------------------------------------------------------

    labels, predictions, spoof_scores = (
        evaluate_model(
            model,
            test_loader,
            device,
        )
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    metrics = calculate_metrics(
        labels,
        predictions,
        spoof_scores,
    )

    print()
    print("=" * 70)
    print("VOICEGUARD — FINAL TEST METRICS")
    print("=" * 70)

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1        : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print(
        f"EER       : {metrics['eer']:.4f}"
    )

    print()
    print("Confusion Matrix")
    print(
        "Rows    : actual"
    )
    print(
        "Columns : predicted"
    )
    print(
        "          BONAFIDE  SPOOF"
    )
    print(
        f"BONAFIDE  {metrics['confusion_matrix'][0, 0]:8d}"
        f"  {metrics['confusion_matrix'][0, 1]:5d}"
    )
    print(
        f"SPOOF     {metrics['confusion_matrix'][1, 0]:8d}"
        f"  {metrics['confusion_matrix'][1, 1]:5d}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()