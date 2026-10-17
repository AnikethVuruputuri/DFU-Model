# DFU-SIAM v3: Multi-Task Explainable AI System for Diabetic Foot Ulcers

Production-ready deep learning system combining **hybrid CNN-Transformer architectures (EfficientNetV2-S + Swin-Tiny)**, **joint multi-task pathology classification & Wagner depth grading**, **disentangled shared/private representations**, and a multi-method **Explainable AI (XAI)** suite (Grad-CAM, Transformer Attribution, Superpixel SHAP, Integrated Gradients).

---

## 🚀 Quickstart: Installation & Setup

### Option 1: Using Pip (Recommended for Lightning AI / Colab / Cloud GPUs)

```bash
# 1. Clone the repository
git clone https://github.com/AnikethVuruputuri/DFU-Model.git
cd DFU-Model

# 2. Install PyTorch with CUDA 12.1 (skip if already installed in your environment)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install required dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda (Local Workstation or Server)

```bash
# 1. Clone repository
git clone https://github.com/AnikethVuruputuri/DFU-Model.git
cd DFU-Model

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate dfu_siam
```

---

## ⚡ Verifying Installation & Hardware

Verify that your GPU and datasets are recognized:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "from dfu_siam_model import DFUSiamModel; m = DFUSiamModel(); print('DFU-SIAM v3 Model initialized successfully!')"
```

---

## 🏋️ Training Models

### 1. Kaggle 2x T4 Parallel Ablations (5.0 - 5.5 Hours Total)
Distributes the 8 ablation experiments across both T4 GPUs in parallel:
```bash
python run_kaggle_ablations_2x_t4.py --epochs 25 --batch_size 16 --workers 4
```

### 2. Run Complete Smart Ablation Suite (Single GPU / Cloud / Cluster)
Automatically runs all ablation experiments with checkpoint resume support and automated summary metrics generation:
```bash
chmod +x run_smart_ablations.sh
nohup ./run_smart_ablations.sh > ablation_runner.log 2>&1 &
tail -f ablation_runner.log
```

### 3. Standalone Training (Full Architecture)
Train with default or custom seeds:
```bash
# Seed 42 (Default)
python train_dfu_siam.py --seed 42 --batch_size 32 --num_workers 8 --epochs 60 --checkpoint_dir checkpoints_seed_42

# Seed 123
python train_dfu_siam.py --seed 123 --batch_size 32 --num_workers 8 --epochs 60 --checkpoint_dir checkpoints_seed_123

# Seed 2026
python train_dfu_siam.py --seed 2026 --batch_size 32 --num_workers 8 --epochs 60 --checkpoint_dir checkpoints_seed_2026
```
*(On NVIDIA A100 / H100 GPUs, you can safely scale `--batch_size 64` and `--num_workers 12` for maximum throughput)*.

---

## 🔍 Explainable AI (XAI) & Single Image Inference

Generate explainability visual heatmaps (Grad-CAM, Transformer Attention, SHAP, Integrated Gradients) for any test image:

```bash
# Single image prediction + XAI explanation
python explain_prediction.py \
  --checkpoint checkpoints_a1_full/best_checkpoint.pth \
  --image DFUdataset/DFUC2021_test/301000.jpg \
  --method all \
  --output_dir outputs/xai/sample_301000
```

---

## 🖥️ Interactive Web & Desktop Apps

### Launch Web UI (Gradio)
Access interactive drag-and-drop diagnostic interface with real-time XAI visualizations from any web browser:
```bash
python app_web.py --port 7860
```

### Launch Desktop GUI (Tkinter)
```bash
python app_gui.py
```

---

## 📁 Repository Structure

```text
├── DFUdataset/                  # DFUC2021 pathology dataset (train, test, labels)
├── WagnerDataset/               # Wagner wound depth dataset (Grades 0 to 4)
├── dfu_siam_model.py            # DFU-SIAM v3 Hybrid Siamese Multi-Task Architecture
├── train_dfu_siam.py            # Multi-loss training pipeline (CORAL, LMCoT, MMD, BYOL)
├── evaluate_checkpoint.py       # Checkpoint evaluation & metric computation
├── run_smart_ablations.sh       # Automated bash runner for all 12 ablation studies
├── explain_prediction.py        # Production XAI inference script
├── xai/                         # Explainable AI modules (Grad-CAM, Rollout, SHAP, IG)
├── app_web.py                   # Gradio Web GUI for interactive diagnostics
├── app_gui.py                   # Desktop GUI application
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip package requirements
└── research_paper/              # LaTeX manuscript, literature review, and audit reports
```
