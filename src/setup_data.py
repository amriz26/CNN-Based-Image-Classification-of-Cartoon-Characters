import os
import shutil
import random
from pathlib import Path

def split_train_val(source_train_dir, target_data_dir, val_ratio=0.15):
    print(f"Splitting {source_train_dir} into train and val...")
    source_path = Path(source_train_dir)
    target_path = Path(target_data_dir)

    for split in ['train', 'val']:
        (target_path / split).mkdir(parents=True, exist_ok=True)

    # Get all categories
    categories = [d for d in source_path.iterdir() if d.is_dir()]

    for category in categories:
        category_name = category.name
        print(f"Processing category: {category_name}")
        
        (target_path / 'train' / category_name).mkdir(parents=True, exist_ok=True)
        (target_path / 'val' / category_name).mkdir(parents=True, exist_ok=True)

        images = list(category.glob('*'))
        # Filter for common image extensions
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        random.shuffle(images)

        n = len(images)
        val_idx = int(n * val_ratio)

        val_imgs = images[:val_idx]
        train_imgs = images[val_idx:]

        for img in train_imgs:
            shutil.copy(img, target_path / 'train' / category_name / img.name)
        for img in val_imgs:
            shutil.copy(img, target_path / 'val' / category_name / img.name)

def copy_test_set(source_test_dir, target_data_dir):
    print(f"Copying {source_test_dir} to test...")
    source_path = Path(source_test_dir)
    target_path = Path(target_data_dir) / 'test'
    target_path.mkdir(parents=True, exist_ok=True)

    # Copy categories in TEST
    categories = [d for d in source_path.iterdir() if d.is_dir()]
    for category in categories:
        category_name = category.name
        print(f"Copying category: {category_name}")
        shutil.copytree(category, target_path / category_name, dirs_exist_ok=True)

if __name__ == "__main__":
    # PROJECT-RELATIVE SETUP:
    # 1. Download the 'Cartoon Classification' dataset from Kaggle
    # 2. Extract it into 'raw_data/downloaded_dataset'
    # 3. You should see 'TRAIN' and 'TEST' folders inside that directory.
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    SOURCE_ROOT = BASE_DIR / "raw_data" / "downloaded_dataset"
    TARGET_DATA_DIR = BASE_DIR / "data"

    if not SOURCE_ROOT.exists():
        print(f"ERROR: Source directory not found at {SOURCE_ROOT}")
        print("Please download the dataset and extract it to 'raw_data/downloaded_dataset' first.")
        print("Link: https://www.kaggle.com/datasets/volkanderekoy/cartoon-classification")
        exit(1)

    # 1. Split TRAIN into train and val
    # The source Kaggle folder has a 'TRAIN' subfolder with category folders
    split_train_val(SOURCE_ROOT / "TRAIN", TARGET_DATA_DIR, val_ratio=0.15)

    # 2. Copy TEST to test
    # The source Kaggle folder has a 'TEST' subfolder with category folders
    copy_test_set(SOURCE_ROOT / "TEST", TARGET_DATA_DIR)

    print("\nData preparation complete!")
    print(f"Your processed data is now in: {TARGET_DATA_DIR}")
