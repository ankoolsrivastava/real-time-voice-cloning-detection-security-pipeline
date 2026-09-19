import torch

from model import VoiceGuardModel


def main():

    print("=" * 70)
    print("VOICEGUARD — MODEL GPU TEST")
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
    # Model
    # --------------------------------------------------

    model = VoiceGuardModel()

    model = model.to(device)

    print(
        "\nModel parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # --------------------------------------------------
    # Fake feature batch
    #
    # This is NOT training.
    #
    # It simply verifies that our architecture
    # accepts the exact Log-Mel shape.
    # --------------------------------------------------

    batch_size = 4

    dummy_features = torch.randn(
        batch_size,
        1,
        64,
        1001,
        device=device,
    )

    labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    logits, attention = model(
        dummy_features
    )

    print("\nInput shape:")
    print(dummy_features.shape)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nAttention shape:")
    print(attention.shape)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels
    )

    print("\nLoss:", loss.item())

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    loss.backward()

    print(
        "\nBackward pass: PASS"
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
            f"GPU memory allocated: {allocated:.2f} MB"
        )

        print(
            f"GPU memory reserved : {reserved:.2f} MB"
        )

    print("\n" + "=" * 70)
    print("FINAL MODEL TEST")
    print("=" * 70)

    if logits.shape != (batch_size, 2):
        raise RuntimeError(
            "Incorrect classifier output shape"
        )

    if attention.ndim != 3:
        raise RuntimeError(
            "Incorrect attention output shape"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Model produced NaN/Inf"
        )

    print(
        "✅ MODEL FORWARD + BACKWARD PASS SUCCESSFUL"
    )


if __name__ == "__main__":
    main()