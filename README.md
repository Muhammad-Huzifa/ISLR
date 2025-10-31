🧠 Sign Language Recognition using ST-GCN and Pose Landmarks

Deep learning-based Sign Language Recognition (SLR) pipeline built using PyTorch and Graph Convolutional Networks (ST-GCN) on human pose landmarks extracted from sign videos.

📁 Project Overview

This repository implements a Spatial-Temporal Graph Convolutional Network (ST-GCN) model trained on pose keypoints (hands, face, body) extracted from sign language videos using MediaPipe.
It supports both 100-class and 300-class configurations from the WLASL dataset.

The workflow includes:

Landmark Extraction from videos

Graph-based Spatio-Temporal Modeling

Training & Evaluation with augmentation and attention modules

⚙️ Project Structure

📦 Spatial-Temporal Graph Convolutional Network (ST-GCN)-Using-Pytorch-and-Tensor-Flow
│
├── src/
│   ├── extract_landmarks.ipynb         # Extracts pose/hand/face landmarks using MediaPipe
│   ├── train_stgcn_100.ipynb           # ST-GCN training for 100 sign classes
│   └── train_stgcn_300.ipynb           # ST-GCN training for 300 sign classes
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Ignore unnecessary files
└── README.md                           

🧩 Features

✅ Pose-based sign representation (no raw video needed)
✅ Graph Convolutional Network (ST-GCN) backbone
✅ Temporal self-attention for motion emphasis
✅ Mixup and label smoothing for regularization
✅ Support for 100-class and 300-class subsets
✅ AMP training for faster GPU computation
✅ Automatic best model checkpoint saving

🧮 Workflow
1️⃣ Extract Landmarks
src/extract_landmarks.ipynb

2️⃣ Train ST-GCN
src/train_stgcn_100.ipynb # for 100 classes
src/train_stgcn_300.ipynb # for 300 classes

🧠 Model Architecture
ST-GCN Block

Each block learns:

Spatial Features: joint-to-joint relations via graph convolution

Temporal Features: motion over time via temporal convolution

Temporal Self-Attention

Focuses on important motion frames, enhancing gesture clarity.

Input (B, 4, 128, 67)
   ↓
ST-GCN × 4
   ↓
Temporal Attention
   ↓
Global Pooling → Linear(256 → Classes)

🧑‍💻 Author

Muhammad Huzaifa

Deep Learning Researcher | Focus: Sign Language Recognition, Transformers, AGI
📧 Contact: mhuzaifa3202@gmail.com





