from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    SAMPLE_RATE,
    MAX_DURATION_SEC,
    NUM_CLASSES,
    GRU_HIDDEN_SIZE,
)

from model import VoiceGuardModel

from audio_preprocessor import (
    VoiceGuardAudioPreprocessor,
)

from feature_extractor import (
    VoiceGuardFeatureExtractor,
)


# ==========================================================
# VOICEGUARD — INFERENCE ENGINE
# ==========================================================

BEST_CHECKPOINT = (
    Path(CHECKPOINT_DIR)
    / "best_model.pt"
)


class VoiceGuardInference:
    """
    VoiceGuard inference engine.

    Pipeline:

        Audio
          ↓
        Preprocessing
          ↓
        Log-Mel
          ↓
        Custom VoiceGuard Model
          ↓
        BONAFIDE / SPOOF
    """

    def __init__(
        self,
        checkpoint_path=BEST_CHECKPOINT,
        device=None,
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                "VoiceGuard trained checkpoint "
                "does not exist yet:\n"
                f"{self.checkpoint_path}\n\n"
                "Final inference is unavailable until "
                "REAL + SPOOF training produces "
                "best_model.pt."
            )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.model = VoiceGuardModel(
            num_classes=NUM_CLASSES,
            hidden_size=GRU_HIDDEN_SIZE,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:

            raise ValueError(
                "Invalid VoiceGuard checkpoint: "
                "'model_state_dict' is missing."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # --------------------------------------------------
        # Same preprocessing pipeline used during training
        # --------------------------------------------------

        self.preprocessor = (
            VoiceGuardAudioPreprocessor(
                target_sr=SAMPLE_RATE,
                max_duration_sec=MAX_DURATION_SEC,
            )
        )

        self.feature_extractor = (
            VoiceGuardFeatureExtractor(
                sample_rate=SAMPLE_RATE,
            )
        )

    def predict(
        self,
        audio_path,
    ):
        """
        Run inference on one real audio file.

        Returns:
            dictionary containing:
                prediction
                bonafide_probability
                spoof_probability
                device
        """

        audio_path = Path(
            audio_path
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if not audio_path.is_file():

            raise ValueError(
                f"Audio path is not a file: {audio_path}"
            )

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        waveform = self.preprocessor.process(
            str(audio_path)
        )

        # --------------------------------------------------
        # Log-Mel
        # --------------------------------------------------

        features = (
            self.feature_extractor
            .extract_with_channel(
                waveform
            )
        )

        if not torch.isfinite(
            features
        ).all():

            raise ValueError(
                "Non-finite Log-Mel features "
                f"detected for {audio_path}"
            )

        # --------------------------------------------------
        # Add batch dimension
        #
        # [1, 64, time]
        #       ↓
        # [1, 1, 64, time]
        # --------------------------------------------------

        features = features.unsqueeze(
            0
        )

        features = features.to(
            self.device
        )

        # --------------------------------------------------
        # Model inference
        # --------------------------------------------------

        with torch.no_grad():

            logits, attention = (
                self.model(features)
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        # --------------------------------------------------
        # Extract probabilities
        # --------------------------------------------------

        bonafide_probability = float(
            probabilities[0, 0]
            .cpu()
            .item()
        )

        spoof_probability = float(
            probabilities[0, 1]
            .cpu()
            .item()
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1,
            )[0]
            .cpu()
            .item()
        )

        if predicted_class == 0:

            prediction = "BONAFIDE"

        else:

            prediction = "SPOOF"

        return {
            "prediction": prediction,

            "bonafide_probability":
                bonafide_probability,

            "spoof_probability":
                spoof_probability,

            "device":
                str(self.device),

            "audio_path":
                str(audio_path),

            "feature_shape":
                list(features.shape),

            "attention_shape":
                list(attention.shape),
        }


# ==========================================================
# SAFE STATUS CHECK
# ==========================================================

def main():

    print("=" * 70)
    print(
        "VOICEGUARD — INFERENCE ENGINE"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )

    if not BEST_CHECKPOINT.exists():

        print()
        print(
            "INFERENCE NOT READY"
        )

        print(
            "Trained VoiceGuard checkpoint "
            "does not exist yet."
        )

        print()
        print(
            "This is expected before REAL + SPOOF "
            "training."
        )

        print()
        print(
            "No dummy model or synthetic prediction "
            "will be created."
        )

        print("=" * 70)

        return

    # ------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------

    engine = VoiceGuardInference(
        checkpoint_path=BEST_CHECKPOINT,
        device=device,
    )

    print()
    print(
        "Trained VoiceGuard model loaded."
    )

    print(
        "Inference engine ready."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()