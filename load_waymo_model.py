import torch
import torch.nn as nn
from pathlib import Path

# Model architecture must match the trained checkpoint
class BEVDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.network(x)


# Select GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint_path = Path("models/waymo_bev_best_model.pth")

model = BEVDetector().to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Waymo BEV model loaded successfully!")
print("Device:", device)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

# Test with sample input
sample = torch.randn(1, 1, 400, 350).to(device)

with torch.no_grad():
    prediction = model(sample)

print("Input shape:", sample.shape)
print("Output shape:", prediction.shape)
print("GPU model test successful!")