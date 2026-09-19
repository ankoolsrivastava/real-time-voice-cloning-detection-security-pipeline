import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    CNN block for extracting local time-frequency patterns.
    """

    def __init__(self, in_channels, out_channels, dropout=0.15):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class TemporalAttention(nn.Module):
    """
    Attention mechanism that learns which temporal
    regions are important for spoof detection.
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x):
        """
        x shape:
            [batch, time, features]

        returns:
            pooled features
            attention weights
        """

        scores = self.attention(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        weighted = x * weights

        pooled = weighted.sum(
            dim=1
        )

        return pooled, weights


class VoiceGuardModel(nn.Module):
    """
    VoiceGuard custom anti-spoofing model.

    Pipeline:

        Log-Mel Spectrogram
                ↓
             CNN
                ↓
             GRU
                ↓
          Attention
                ↓
          Classifier
                ↓
       BONAFIDE / SPOOF
    """

    def __init__(
        self,
        num_classes=2,
        hidden_size=128,
    ):
        super().__init__()

        # --------------------------------------------------
        # CNN
        # --------------------------------------------------

        self.cnn = nn.Sequential(

            ConvBlock(
                1,
                32,
                dropout=0.10
            ),

            ConvBlock(
                32,
                64,
                dropout=0.15
            ),

            ConvBlock(
                64,
                128,
                dropout=0.20
            ),

            ConvBlock(
                128,
                128,
                dropout=0.20
            ),
        )

        # --------------------------------------------------
        # Frequency aggregation
        # --------------------------------------------------

        self.frequency_pool = nn.AdaptiveAvgPool2d(
            (1, None)
        )

        # --------------------------------------------------
        # GRU
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )

        # Bidirectional GRU doubles feature size.
        gru_features = hidden_size * 2

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        self.attention = TemporalAttention(
            gru_features
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                gru_features,
                128
            ),

            nn.BatchNorm1d(128),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                num_classes
            ),
        )

    def forward(self, x):
        """
        Input:

            [batch, 1, 64, time]

        Output:

            logits:
                [batch, 2]

            attention_weights:
                [batch, time, 1]
        """

        # --------------------------------------------------
        # CNN
        # --------------------------------------------------

        x = self.cnn(x)

        # --------------------------------------------------
        # Pool frequency dimension
        #
        # [B, C, F, T]
        #      ↓
        # [B, C, 1, T]
        # --------------------------------------------------

        x = self.frequency_pool(x)

        # Remove frequency dimension

        x = x.squeeze(2)

        # [B, C, T]
        #
        # GRU expects:
        # [B, T, C]

        x = x.transpose(1, 2)

        # --------------------------------------------------
        # GRU
        # --------------------------------------------------

        x, _ = self.gru(x)

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        x, attention_weights = self.attention(
            x
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        logits = self.classifier(
            x
        )

        return logits, attention_weights