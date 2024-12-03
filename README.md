# MNIST Classification with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification, optimized for high accuracy (>99.4%) with less than 20K parameters.

## Model Architecture

The model uses a lightweight CNN architecture with the following key features:

- **Total Parameters**: ~13,010
- **Architectural Features**:
  - Batch Normalization for better training stability
  - Dropout layers for regularization (0.1)
  - Global Average Pooling (GAP) to reduce parameters
  - Strategic use of 1x1 convolutions
  - No fully connected layers

### Layer Structure

python

Input (1, 28, 28)

│

├── Block 1

│ ├── Conv2d(1 → 8, 3x3) + BN + ReLU

│ ├── Conv2d(8 → 16, 3x3) + BN + ReLU

│ ├── MaxPool2d(2x2)

│ └── Dropout(0.1)

│

├── Block 2

│ ├── Conv2d(16 → 16, 3x3) + BN + ReLU

│ ├── Conv2d(16 → 24, 3x3) + BN + ReLU

│ ├── MaxPool2d(2x2)

│ └── Dropout(0.1)

│

├── Block 3

│ ├── Conv2d(24 → 24, 3x3) + BN + ReLU

│ ├── Conv2d(24 → 16, 1x1) + BN + ReLU

│ ├── Global Average Pooling

│ └── Conv2d(16 → 10, 1x1)

│

└── Output (10)

## Requirements

bash

torch>=2.0.0

torchvision>=0.15.0

tqdm>=4.65.0

numpy>=2.0.0

matplotlib>=3.7.0

pillow>=9.5.0

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── model.py           # Model architecture definition
├── train.py          # Training and evaluation code
├── requirements.txt  # Project dependencies
├── .github
│   └── workflows
│       └── model_checks.yml  # CI pipeline for model verification
└── README.md
```

## Training Configuration

- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rate**: 0.01 with OneCycleLR scheduler
  - Division factors: 10 (initial), 100 (final)
- **Batch Size**: 128
- **Epochs**: 10
- **Data Augmentation**:
  - Random rotation (±15°)
  - Random affine transforms
    - Translation: up to 10-20%
    - Scale: ±10%
    - Shear: ±10°
  - Normalization (mean=0.1307, std=0.3081)

## Parameter Distribution

Layer-wise parameter breakdown:

- First Block:

  - Conv1: (3×3×1×8) + 8 = 80 parameters
  - BN1: 16 parameters
  - Conv2: (3×3×8×16) + 16 = 1,168 parameters
  - BN2: 32 parameters
- Second Block:

  - Conv3: (3×3×16×16) + 16 = 2,320 parameters
  - BN3: 32 parameters
  - Conv4: (3×3×16×24) + 24 = 3,480 parameters
  - BN4: 48 parameters
- Third Block:

  - Conv5: (3×3×24×24) + 24 = 5,184 parameters
  - BN5: 48 parameters
  - Conv6: (1×1×24×16) + 16 = 400 parameters
  - BN6: 32 parameters
  - Conv7: (1×1×16×10) + 10 = 170 parameters

Total: ~13,010 parameters

## Model Verification

The repository includes automated tests (GitHub Actions) that verify:

1. Parameter count (< 20K)
2. Use of Batch Normalization
3. Use of Dropout
4. Use of Global Average Pooling
5. Correct output shape
6. Forward pass functionality

## Usage

1. **Training the model**:

```bash
python train.py
```

2. **Model checkpoints**:

- Best model saved automatically during training
- Checkpoints saved in `checkpoints/` directory
- Format: `model_epoch_X_acc_YY.YY.pth`
- Final model saved in `checkpoints/final/`

3. **Parameter counting**:

```python
from model import Net
model = Net()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')
```

## Model Loading

```python
# Load saved model
checkpoint = torch.load('checkpoints/final/model_epoch_9.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Continuous Integration

GitHub Actions workflow (`model_checks.yml`) automatically verifies:

- Model architecture requirements
- Parameter constraints
- Basic functionality

Tests run on:

- Every push to main branch
- Every pull request to main branch

## Acknowledgments

- PyTorch framework
- MNIST dataset creators
