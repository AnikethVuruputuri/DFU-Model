# DFU-SIAM v3 Reproducibility Checklist & Audit

This checklist provides an item-by-item verification of the code, data manifests, dependencies, and configuration artifacts required for independent computational reproduction.

---

## 1. Itemized Reproducibility Audit

| Checklist Item | Status | Location / Artifact | Verification Details |
| :--- | :---: | :--- | :--- |
| **Random Seeds** | ✅ PASS | `experiment_config.json`, `train_dfu_siam.py` | Fixed seed `42` configured for PyTorch, NumPy, Python random, and pair generation. |
| **Deterministic Mode Flag** | ⚠️ PARTIAL | `experiment_config.json` | `--deterministic` flag exists, but was set to `false` for CUDA CuDNN benchmarking speed. |
| **Configuration Files** | ✅ PASS | `checkpoints_a1_full_balanced/experiment_config.json` | Exact JSON dump of all 59 hyperparameters saved at run initialization. |
| **Dataset Manifests** | ✅ PASS | `temp_train_split.csv` & `temp_val_split.csv` | Exact 80/20 train (4,764 samples) and val (1,191 samples) split manifests preserved in checkpoint directory. |
| **Deduplication Manifests** | ✅ PASS | `wagner_dedup_manifest.json` | Hash-based manifest excluding 148 leaking samples (37 exact duplicates + 111 augmented variants). |
| **Environment & Package Versions** | ✅ PASS | `requirements.txt`, `experiment_config.json` | Explicitly records Python 3.11.15, PyTorch 2.6.0+cu124, timm 1.0.15, scikit-learn 1.6.1, shap 0.51.0. |
| **Git Commit / Version Tracking** | ❌ FAIL | Workspace root | Repository root is **not a git repository** (`fatal: not a git repository`). No commit hashes recorded. |
| **Checkpoint Loading & Integrity** | ✅ PASS | `evaluate_checkpoint.py`, `test_xai.py` | Verified loadable; exact state dict match; architecture parameter consistency confirmed. |
| **Resume Training Mechanism** | ✅ PASS | `train_dfu_siam.py` | Atomic saving via `.tmp` swap; RNG states, optimizer states, AMP scaler, and epoch counters preserved. |
| **Exact Preprocessing Protocol** | ✅ PASS | `dataset.py` (`get_dfu_transforms`) | Resize(224, 224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). |
| **Exact Data Augmentation** | ✅ PASS | `dataset.py` (`clinical` level) | RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.2), RandomRotation(15), ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1). |
| **Exact Optimizer Specification** | ✅ PASS | `train_dfu_siam.py`, `experiment_config.json` | AdamW with differential learning rate: `lr_backbone=1e-5`, `lr_head=2e-4`, `weight_decay=1e-3`. |
| **Exact Learning Rate Scheduler** | ✅ PASS | `train_dfu_siam.py` | CosineAnnealingLR with 3-epoch linear warmup; minimum LR = 1e-7. |
| **Batch Size & Pair Count** | ✅ PASS | `experiment_config.json` | Batch size = 8; 6,000 multi-task Siamese pairs per epoch. |
| **Epoch Schedule & Stopping** | ⚠️ PARTIAL | `checkpoints_a1_full_balanced/` | Target was 45 epochs; run was paused/saved at Epoch 36 (`best_checkpoint.pth`). |
| **Hardware Specification** | ✅ PASS | System environment & `benchmark_lightweight.py` | Host: Windows 11 AMD64; GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, 2,048 CUDA cores). |

---

## 2. Overall Reproducibility Assessment

### Reproducibility Score: **7.5 / 10**

### Justification:
* **Strengths (+7.5)**: The repository maintains exceptionally thorough configuration dumps (`experiment_config.json`), exact train/validation CSV split manifests, deduplication exclusion manifests (`wagner_dedup_manifest.json`), atomic checkpoint reloadability, and unit test verification scripts (`test_all.py`, `test_xai.py`). Any researcher with access to the DFUC2021 and Wagner raw images can recreate the identical data loaders and model configuration.
* **Weaknesses (-2.5)**:
  1. **Absence of Version Control (-1.5)**: The project directory lacks an initialized Git repository, preventing cryptographic tracking of commit history and code diffs.
  2. **Single-Seed Execution (-1.0)**: The balanced model was trained only on `seed 42`; multi-seed variance scripts are implemented but have not been executed.
