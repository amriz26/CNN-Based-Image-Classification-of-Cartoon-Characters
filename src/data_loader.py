import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=32):
    """
    Creates DataLoaders for train, val, and test splits.
    
    Expected directory structure:
    data_dir/
        train/
            class1/
            class2/
        val/
            class1/
            class2/
        test/
            class1/
            class2/
    """
    
    # Standard ImageNet normalization constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    
    # Create DataLoaders
    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    # Get dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    return data_loaders, dataset_sizes, class_names

if __name__ == "__main__":
    # Test block
    DATA_DIR = "data"
    
    if os.path.exists(DATA_DIR):
        try:
            loaders, sizes, classes = get_data_loaders(DATA_DIR)
            print(f"Classes: {classes}")
            print(f"Dataset sizes: {sizes}")
            
            # Check a single batch
            train_loader = loaders['train']
            images, labels = next(iter(train_loader))
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print(f"Data directory '{DATA_DIR}' not found. Please run setup_data.py first.")
