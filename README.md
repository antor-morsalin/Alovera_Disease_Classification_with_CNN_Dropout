# Aloe Vera Leaf Disease Classification with CNN Dropout

A computer vision experiment investigating how dropout affects the learning behavior and generalization of a custom convolutional neural network (CNN) for aloe vera leaf condition classification.

## Dataset

The experiment uses the [Aloe Vera Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/morsalinantor/aloe-vera-leaf-disease-detection-dataset), containing 2,500 images across five classes.

| Class | Images |
| --- | ---: |
| Aloe Rust | 210 |
| Anthracnose | 500 |
| Healthy | 780 |
| Leaf Spot | 500 |
| Sunburn | 510 |

Images are resized to 224 × 224 pixels. Training augmentation includes horizontal and vertical flips, rotation, and color jitter. The intended stratified split is 70% training, 15% validation, and 15% testing.

## Model Architecture

The custom `AloeCNN` extracts spatial features through five convolutional layers, followed by a fully connected classifier.

```text
RGB image (3 × 224 × 224)
  → Convolution + ReLU + MaxPool (32 channels)
  → Convolution + ReLU + MaxPool (128 channels)
  → Convolution + ReLU + MaxPool (256 channels)
  → Convolution + ReLU + MaxPool (128 channels)
  → Convolution + ReLU (16 channels)
  → Dropout → Flatten → Linear (3136 → 256)
  → ReLU → Dropout → Linear (256 → 5)
```

## Experimental Configuration

| Setting | Baseline | Dropout model |
| --- | ---: | ---: |
| Feature dropout | 0 | 0.15 |
| Hidden-layer dropout | 0 | 0.50 |
| Epochs | 100 | 150 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Loss | Cross-entropy | Cross-entropy |

## Evaluation

The notebook records and plots training and validation accuracy and loss. The comparison explores whether dropout reduces overfitting.

**Evaluation caveat:** The current notebook uses indices relative to the temporary subset directly on the full dataset, causing possible training–evaluation overlap. The split must be corrected and both models retrained before reporting reliable held-out results. Equal training budgets and multiple seeds would also strengthen the comparison.

## Getting Started

```bash
git clone https://github.com/antor-morsalin/Alovera_Disease_Classification_with_CNN_Dropout.git
cd Alovera_Disease_Classification_with_CNN_Dropout
pip install torch torchvision fastai numpy scikit-learn matplotlib jupyter
jupyter notebook cvpr-mid-assignment.ipynb
```

The notebook was developed on Kaggle. Attach the dataset there or adapt the input and output paths for local execution. GPU acceleration is recommended.

## Author

**Mahmudul Morsalin Antor**

[GitHub](https://github.com/antor-morsalin) · [Portfolio](https://morsalin-antor-portfolio.vercel.app)
