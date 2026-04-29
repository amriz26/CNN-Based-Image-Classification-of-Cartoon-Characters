from src.data import get_dataloaders
import torch

def main():
    print("--- Testing Modular Data Pipeline ---")
    
    # Plug-and-play: No arguments needed because we set a default data path!
    try:
        train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=32)
        
        print(f"Dataset Classes: {class_names}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a single batch
        images, labels = next(iter(train_loader))
        print(f"\nTraining Batch Shape: {images.shape}")
        print(f"Successfully loaded {len(images)} images.")
        
        print("\nPipeline check successful! You can now use this import in your training script.")
        
    except Exception as e:
        print(f"Pipeline check failed: {e}")
        print("Tip: Run 'python src/setup_data.py' to generate the 'data/' folder from your raw images.")

if __name__ == "__main__":
    main()
