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
🚀 Quick Start
1. Installation
Clone the repo and install dependencies:
