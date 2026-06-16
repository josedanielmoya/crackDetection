# Concrete Crack Detection using Fine-Tuned ResNet50

Automated detection of cracks in concrete surfaces via Transfer Learning on the Maguire dataset.

---

## Project Overview

This project implements a binary image classifier to distinguish cracked from non-cracked concrete surfaces. The main challenges were a heavily imbalanced dataset (1:5.6 ratio) and the need for high recall on the minority class (cracks).

The approach combines dynamic class weighting with a two-stage transfer learning strategy on ResNet50: first training a frozen backbone, then fine-tuning the full network at a reduced learning rate.

---

## Repository Structure

```
crackDetection/
│
├── Article___Automated_crack_detection_in_concrete_surfaces.pdf
│
└── notebooks/
    ├── 01_data_preparation.ipynb
    ├── 02_training_v2_save_no_finetune.ipynb
    └── 03_training_final_finetuned.ipynb
```

- `01_data_preparation.ipynb` — Walks through the initial setup of the dataset. The raw Maguire images are spread across multiple subfolders (pavement, wall, deck — cracked and uncracked variants). This notebook consolidates all of them into two unified folders (`Cracked/` and `Non_Cracked/`), disambiguating filenames in the process. It then loads the full dataset using TensorFlow's `image_dataset_from_directory`, applies an 80/20 train/validation split, and computes dynamic class weights to compensate for the 1:5.6 class imbalance. In-memory caching and asynchronous prefetching are configured to speed up training.

- `02_training_v2_save_no_finetune.ipynb` — Implements Stage 1 of the transfer learning pipeline. A ResNet50 backbone pre-trained on ImageNet is loaded with its weights frozen, and a custom classification head is attached (GlobalAveragePooling2D, Dropout at 0.2, and a single sigmoid output). A data augmentation block (random flips, rotations, and contrast shifts) is applied during training. The model is trained for 10 epochs using the Adam optimizer at `lr=0.001` with Binary Crossentropy loss and class weights applied. After training, performance is evaluated on the validation set at two decision thresholds (0.5 and 0.3) and the model is saved to disk.

- `03_training_final_finetuned.ipynb` — Implements Stage 2 by unfreezing the full ResNet50 backbone and continuing training for 5 additional epochs at a much lower learning rate (`lr=1e-5`) to avoid catastrophic forgetting of ImageNet features. The fine-tuned model is then evaluated on the validation set at two thresholds (0.5 for balanced performance, 0.8 to maximize crack recall at the cost of precision), and a confusion matrix is produced. This notebook contains the final results reported in the project.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Maguire Dataset (2018) |
| Total images | 56,092 |
| Cracked | 8,484 |
| Non_Cracked | 47,608 |
| Train / Val split | 80% / 20% |
| Image size | 224 x 224 px |
| Class weight (Cracked) | 3.31 |
| Class weight (Non_Cracked) | 0.59 |

---

## Results

**Stage 1 — Frozen backbone (10 epochs)**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cracked | 0.74 | 0.63 | 0.68 |
| Non_Cracked | 0.94 | 0.96 | 0.95 |

**Stage 2 — Fine-tuned (5 epochs, threshold = 0.5)**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cracked | 0.79 | 0.78 | 0.78 |
| Non_Cracked | 0.96 | 0.96 | 0.96 |

---

## Tech Stack

- TensorFlow 2.10 / Keras
- scikit-learn
- Matplotlib, Seaborn
- NumPy

---

## How to Reproduce

1. Download the [Maguire 2018 dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2) and update the `source_dir` path in `01_data_preparation.ipynb`.
2. Run the notebooks in order.
3. A CUDA-compatible GPU is recommended.

---

## Reference

See `Article___Automated_crack_detection_in_concrete_surfaces.pdf` for the paper this project is based on.
