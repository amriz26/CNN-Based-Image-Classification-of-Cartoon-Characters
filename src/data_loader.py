import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os

class ApplyTransform(Dataset):
    """
    A small wrapper to apply specific transforms to a Subset of a dataset.
    This allows different transforms (like augmentation) for train vs val/test
    even when they come from the same base dataset.
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
    Creates DataLoaders for train, val, and test splits using random_split.
    
    Args:
        data_dir: Path to the root directory containing all classes.
        batch_size: Number of samples per batch.
        seed: Random seed for reproducibility.
    """
    
    # Standard ImageNet normalization constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Define transforms
    # For now, we use the same simple transforms for all splits as requested.
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load the entire dataset (without transforms initially to allow flexibility)
    full_dataset = datasets.ImageFolder(data_dir)
    total_size = len(full_dataset)
    
    # Calculate split sizes (70%, 15%, 15%)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total dataset size: {total_size}")
    print(f"Splitting into Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split the dataset with a fixed seed
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Wrap subsets with transforms
    train_dataset = ApplyTransform(train_subset, transform=base_transform)
    val_dataset = ApplyTransform(val_subset, transform=base_transform)
    test_dataset = ApplyTransform(test_subset, transform=base_transform)
    
    # Create DataLoaders
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
    # Test block
    # Note: Consolidated data is now in 'raw_data/all_images'
    DATA_DIR = "raw_data/all_images"
    
    if os.path.exists(DATA_DIR):
        try:
            loaders, sizes, classes = get_data_loaders(DATA_DIR)
            print(f"Classes: {classes}")
            print(f"Dataset sizes: {sizes}")
            
            # Check a single batch from training
            train_loader = loaders['train']
            images, labels = next(iter(train_loader))
            print(f"\nBatch inspection:")
            print(f"Images batch shape: {images.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Successfully loaded a batch of {len(images)} images.")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print(f"Data directory '{DATA_DIR}' not found. Please ensure data is consolidated first.")
