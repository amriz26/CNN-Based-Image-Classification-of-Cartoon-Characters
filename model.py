import torch
import torch.nn as nn


class CartoonCNN(nn.Module):
    def __init__(self, num_classes, input_size=(3, 128, 128)):
        """
        Args:
            num_classes (int): Number of output classes, passed in dynamically
                               from the DataLoader so it never needs to be hard-coded.
            input_size  (tuple): (C, H, W) of the images fed into the model.
                                 Used to compute flattened feature dimensions
                                 automatically — changing image resolution here
                                 is all that is needed; no manual arithmetic required.
        """
        super(CartoonCNN, self).__init__()

        # --- Feature extractor: 3 convolutional blocks ---
        # Each block follows the Conv → ReLU → MaxPool pattern.
        # Channels double every block (32 → 64 → 128) to capture
        # increasingly abstract representations.
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # --- Dynamic feature-size calculation ---
        # Instead of hard-coding the flattened dimension (e.g., 32768), we run
        # a single dummy tensor through self.features at init time to let PyTorch
        # compute the exact output shape. This means the classifier stays correct
        # even if input_size is changed, with zero manual recalculation.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            n_features = self.features(dummy).numel()

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Checkpoint utility
# ---------------------------------------------------------------------------

def load_checkpoint(path, num_classes, input_size=(3, 128, 128)):
    """
    Load a saved model checkpoint safely onto the correct device.

    Args:
        path        (str):   Path to the .pth file produced by torch.save().
        num_classes (int):   Must match the value used when the model was trained.
        input_size  (tuple): (C, H, W) used during training (default 128×128 RGB).

    Returns:
        model: CartoonCNN instance in eval() mode, already moved to the
               available device (CUDA if present, otherwise CPU).

    Why map_location matters:
        torch.load() re-maps tensor storage to the device string passed via
        map_location. Without it, a checkpoint saved on a GPU machine will
        attempt to allocate CUDA tensors immediately on load — crashing any
        CPU-only environment with a RuntimeError.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CartoonCNN(num_classes=num_classes, input_size=input_size)

    # map_location ensures tensors are re-mapped to `device` regardless of
    # where they were originally saved.
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # Set to eval mode; no dropout/BN updates needed for inference.

    print(f"Checkpoint loaded from '{path}' onto {device}.")
    return model