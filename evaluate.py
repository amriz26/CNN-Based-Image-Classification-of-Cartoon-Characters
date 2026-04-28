import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import CartoonCNN, load_checkpoint
from src.data import get_dataloaders

# Configuration
# CHECKPOINT_PATH: wherever best_model lives
# BATCH_SIZE: keep at 32 to match training
# VAL_LOSSES: paste in the list from train.ipynb

CHECKPOINT_PATH = "best_model.pth"
BATCH_SIZE = 32
VAL_LOSSES = []

# Load data and model. Returns all three splits plus the class name list
_, _, test_loader, class_names = get_dataloaders(batch_size=BATCH_SIZE)
num_classes = len(class_names)

# builds a fresh CartoonCNN, loads the saved weights
model = load_checkpoint(CHECKPOINT_PATH, num_classes=num_classes)
# See whichever device the model is on
device = next(model.parameters()).device

# Iterate through every batch in test_loader, run a forward pass, and 
# collect all predictions and labels into flat lists
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        # Raw logits: shape(batch, num_classes)
        preds = torch.argmax(outputs, dim=1)
        # Predicted class index per image
        all_preds.extend(preds.cpu().numpy)
        # Move back to CPU before converting
        all_labels.extend(labels.numpy())

# Overall accuracy. Count how many predictions match the ground truth,
# divide by total
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

overall_accuracy = (all_preds == all_labels).mean() * 100
print(f"\n{'='*50}")
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print(f"{'='*50}\n")

# Per-Class accuracy
# Overall accuracy hides problems. A model that's great at SpongeBob but
# is terrible at Tsubasa still looks ok in the overall number
# Per class accuracy reveals exactly where the model is struggling
# For each class, we find all images that truly belong to the class,
# then check what fraction of those the model predicted correctly
print("Per-Class Accuracy:")
print("-" * 35)
for i, name in enumerate(class_names):
    class_mask = (all_labels == i)
    class_total = class_mask.sum()
    class_correct = (all_preds[class_mask] == i).sum()
    class_accuracy = (class_correct / class_total) * 100
    print(f". {name:<20} {class_accuracy:.2f}% ({class_correct}/{class_total})")
    print()

# Confusion Matrix
"""
Should show:
- Rows = true class
- Columns = predicted class
- Diagonal = correct predictions
- Off-Diagonal = mistakes 

"""
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title("Confusion Matrix - Test Set", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png\n")


# Plotting training and validation loss over epochs
"""
Should tell us:
- If the model is learning
- Is it overfitting?
- Did the training converge?

"""
if VAL_LOSSES:
    epochs = range(1, len(VAL_LOSSES) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, VAL_LOSSES, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
    print("Loss curve saved to loss_curve.png\n")
else:
    print("VAL_LOSSES is empty. Skipping loss curve")