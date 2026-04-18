import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os

class ApplyTransform(Dataset):
    """
    A small wrapper to apply specific transforms to a Subset of a dataset.
    This allows different transforms (like augmentation for training) vs 
    standard transforms (val/test) even when they come from the same base dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_data_loaders(data_dir, batch_size=32, seed=42):
    """
    Creates DataLoaders for train, val, and test splits with data augmentation for training.
    
    Args:
        data_dir: Path to the root directory containing all classes.
        batch_size: Number of samples per batch.
        seed: Random seed for reproducibility (Step 2).
    """
    
    # Standard ImageNet normalization constants
    # Normalizing images helps the model converge faster by keeping gradients stable.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # --- STEP 3: DATA AUGMENTATION (Training only) ---
    # Augmentation creates 'new' versions of images by applying random changes.
    # This prevents the model from memorizing specific images and helps it learn general patterns.
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),            # Standardize image size (Step 1)
        transforms.RandomHorizontalFlip(p=0.5),   # 50% chance to flip horizontally
        transforms.RandomRotation(10),            # Random rotation within +/- 10 degrees
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, 
            saturation=0.2, hue=0.1
        ),                                        # Random variations in lighting and color
        transforms.ToTensor(),                    # Convert image pixels to [0, 1] range (Step 1)
        transforms.Normalize(mean, std)           # Adjust pixels to shared mean/std (Step 1)
    ])
    
    # --- VALIDATION & TEST TRANSFORMS (Simple) ---
    # We do NOT augment val/test. We want these to accurately represent real-world data.
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # STEP 1: Load the entire dataset
    # We load without transforms initially so we can apply different ones to each subset later.
    full_dataset = datasets.ImageFolder(data_dir)
    total_size = len(full_dataset)
    
    # STEP 2: Calculate split sizes (70% Train, 15% Val, 15% Test)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total dataset size: {total_size}")
    print(f"Splitting into Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # STEP 2: Split the dataset with a fixed seed for reproducibility.
    # This ensures that 'train' and 'test' images don't mix between different runs.
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Wrap subsets with their specific transforms
    train_dataset = ApplyTransform(train_subset, transform=train_transform)
    val_dataset = ApplyTransform(val_subset, transform=val_test_transform)
    test_dataset = ApplyTransform(test_subset, transform=val_test_transform)
    
    # STEP 1: Create DataLoaders
    # Shuffle=True for training is critical so the model doesn't learn the order of the data.
    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    class_names = full_dataset.classes
    
    return data_loaders, dataset_sizes, class_names

if __name__ == "__main__":
    # Verify the updated pipeline
    DATA_DIR = "raw_data/all_images"
    
    if os.path.exists(DATA_DIR):
        try:
            print("--- Testing Data Augmentation Pipeline ---")
            loaders, sizes, classes = get_data_loaders(DATA_DIR)
            print(f"Classes found: {classes}")
            
            # Load one batch from the training loader
            train_loader = loaders['train']
            images, labels = next(iter(train_loader))
            
            print(f"\nBatch Information:")
            print(f"Images batch shape: {images.shape} (Batch Size, Channels, Height, Width)")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Successfully loaded a batch of {len(images)} augmented images.")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print(f"Directory '{DATA_DIR}' not found. Please consolidate your images first.")
