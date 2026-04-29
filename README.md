# 🎨 Cartoon Character Image Classification

A convolutional neural network (CNN) implemented in PyTorch for classifying cartoon images into their correct categories or shows. Designed to learn visual patterns like color, shape, and texture for accurate image classification in a collaborative deep learning project. . The project focuses on image feature learning through convolutional layers for accurate multi-class image classification

## 🚀 Quick Start for Team Members

To get started with this project, follow these steps to set up your local environment and data pipeline.

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition
We used the **Cartoon Classification** dataset from Kaggle.
*   **Source Dataset:** [https://www.kaggle.com/datasets/volkanderekoy/cartoon-classification](https://www.kaggle.com/datasets/volkanderekoy/cartoon-classification)
*   **Download Instructions:**
    1.  Download the dataset as a ZIP file from the link above.
    2.  Extract the ZIP.
    3.  Move the extracted content into the project's `raw_data/downloaded_dataset/` folder.
    
    Your directory structure should look like this:
    ```text
    raw_data/
    └── downloaded_dataset/
        ├── TRAIN/
        └── TEST/
    ```

### 3. Data Consolidation & Pre-processing
Run the setup script to split the data into training, validation, and test sets according to our project's modular pipeline:
```bash
python src/setup_data.py
```
This will create a `data/` directory with processed images ready for training.

---

## 🛠 Project Structure

*   `src/data/`: Modular data pipeline (transforms, datasets, and dataloaders).
*   `src/setup_data.py`: Script to prepare and split the raw dataset.
*   `test_data_pipeline.py`: Utility to verify that the pipeline is working correctly.
*   `data/`: (Auto-generated) Contains the processed `train/`, `val/`, and `test/` splits.
*   `raw_data/`: (Local only) Container for the raw downloaded dataset (e.g., `downloaded_dataset/TRAIN`).

## 🧪 Testing the Pipeline
After setting up the data, you can verify everything is working by running:
```bash
python test_data_pipeline.py
```

---

## 📝 Collaborative Notes
*   **Git Policy:** The `data/` and `raw_data/` folders are ignored by Git to avoid bloat. Never commit large image files.
*   **Kaggle API:** For automated downloads, ensure you have your `kaggle.json` credentials configured. You can use the `kaggle` CLI (installed via `requirements.txt`) to download the dataset.
