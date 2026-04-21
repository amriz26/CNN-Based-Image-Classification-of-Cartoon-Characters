import torch
import torch.nn as nn
import torch.optim as optim
from model import CartoonCNN, load_checkpoint
from src.data import get_dataloaders
import os

def run_sanity_check():
    print("🚀 Starting Sanity Check...")
    
    # 1. Setup Data
    # Note: Using small batch_size and limited iterations for demonstration
    batch_size = 32
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir="data/train", # Pointing to actual data
        batch_size=batch_size,
        num_workers=0 # Lower workers for stability in this environment
    )
    num_classes = len(class_names)
    print(f"✅ Data loaded. Classes: {num_classes} ({', '.join(class_names[:5])}...)")

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CartoonCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"✅ Model initialized on {device}.")

    # 3. Train for a few batches (simulating 1 epoch)
    model.train()
    print("\n--- Training (Simulated 1 Epoch, 10 Batches) ---")
    limit = 10
    for i, (images, labels) in enumerate(train_loader):
        if i >= limit: break
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2 == 0:
            print(f"Batch [{i+1}/{limit}] | Loss: {loss.item():.4f}")

    # 4. Save Checkpoint
    checkpoint_path = 'model_checkpoint_test.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Checkpoint saved to {checkpoint_path}")

    # 5. Verify Checkpoint Loading
    print("\n--- Checkpoint Verification ---")
    loaded_model = load_checkpoint(checkpoint_path, num_classes=num_classes)
    
    # 6. Compare Outputs
    model.eval()
    loaded_model.eval()
    
    # Get a sample batch
    images, labels = next(iter(val_loader))
    images = images.to(device)
    
    with torch.no_grad():
        original_output = model(images)
        loaded_output = loaded_model(images)
    
    # Numerical comparison
    diff = torch.abs(original_output - loaded_output).max().item()
    print(f"Max difference between original and loaded outputs: {diff:.8f}")
    
    if diff < 1e-6:
        print("🎉 SUCCESS: The model loaded back correctly! Outputs are identical.")
    else:
        print("❌ FAILURE: Discrepancy detected in model outputs.")

    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🧹 Cleaned up {checkpoint_path}")

if __name__ == "__main__":
    run_sanity_check()
