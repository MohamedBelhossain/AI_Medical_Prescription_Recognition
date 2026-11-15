"""Optional CRNN/CTC baseline for handwritten word recognition."""

from __future__ import annotations

try:
    import torch
    from torch import nn

    class CRNN(nn.Module):
        """Compact convolutional recurrent network returning CTC logits."""

        def __init__(self, num_classes: int, hidden_size: int = 256) -> None:
            super().__init__()
            if num_classes < 2:
                raise ValueError("num_classes must include blank and at least one character")
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((1, None)),
            )
            self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, dropout=0.1)
            self.classifier = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.features(images).squeeze(2).permute(2, 0, 1)
            sequence, _ = self.rnn(features)
            return self.classifier(sequence).log_softmax(2)

except ImportError:  # Keep data and preprocessing utilities usable without PyTorch.
    class CRNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("CRNN requires PyTorch. Install project dependencies first.")
