# NTIRE 2026: Day & Night Dual-Focused Raindrop Removal

This repository contains the PyTorch implementation for the **NTIRE 2026 Challenge on Image Restoration (Raindrop Removal)**. Our approach leverages a Transformer-based architecture (Restormer) to handle both "Drop-Focused" and "Background-Focused" degradations across day and night domains.

## 📂 Dataset Structure
The code expects the NTIRE dataset to be organized as follows:
```text
Dataset/
├── Daytime/
│   ├── drop/   # Input: Focused on raindrops (background blurry)
│   ├── blur/   # Input: Focused on background (raindrops blurry)
│   └── clear/  # Ground Truth
└── Nighttime/
    ├── drop/
    ├── blur/
    └── clear/
```

##🚀 Quick Start
#1. Installation
Clone the repo and install dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/NTIRE2026_Raindrop.git](https://github.com/YOUR_USERNAME/NTIRE2026_Raindrop.git)
cd NTIRE2026_Raindrop
pip install -r requirements.txt
```
#2. Training
To train the model from scratch:
```bash
python train.py
```
Note: You can adjust batch size, patch size, and learning rate in the CONFIG dictionary inside train.py.
#3. Validation / Inference
To generate images for the Codabench server (406 mixed validation images):

Place your validation inputs in a folder (e.g., ./Val_Input).

Run the inference script:
`python val_codabench.py`
This script automatically handles:

Reflective Padding: Ensures image dimensions are multiples of 8 (required for Transformers).

Dimension Recovery: Crops the output back to original resolution.

Normalization: Clamps values to [0, 1].

#📊 Evaluation Metrics
The competition uses a composite metric emphasizing Y-channel fidelity and perceptual quality:
$$ Score = PSNR_Y + 10 \times SSIM_Y - 5 \times LPIPS $$

We implement this strict evaluation in utils/metrics.py to track best checkpoints during training.

#📝 Acknowledgements
Restormer: [Zamir et al., CVPR 2022]

NTIRE 2026 Organizers for the dataset.
