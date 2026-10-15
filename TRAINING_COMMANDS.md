# DFU-SIAM v3: Complete Training & Reproduction Command Reference

This document provides copy-paste ready commands for training, ablations, benchmark pilots, checkpoint resumption, and standalone evaluation of the **DFU-SIAM v3** architecture on Windows PowerShell.

---

## 0. Environment Setup & Verification

Before starting training, verify your environment and confirm that all 16 integrity and architecture unit tests pass.

```powershell
# Activate Conda Environment (adjust if needed)
# conda activate ABFormer

# Run 16-point comprehensive test suite
python test_all.py
```

---

## 1. Quick Pilot / Sanity Check Commands

Use these fast sanity commands to verify data loading, GPU memory allocation, backward autograd pass, and checkpointing without running a full training schedule.

### 1.1 Dry Run (Configuration check only, 0 epochs)
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --dry_run
```

### 1.2 Rapid 2-Epoch Pilot (Swin-Tiny 35.2M, ~1 minute)
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 2 `
  --max_batches_per_epoch 8 `
  --checkpoint_every 4 `
  --output_dir checkpoints_pilot `
  --seed 42
```

---

## 2. Core Research Architectures (40 Epochs)

### 2.1 Swin-Tiny Proposed Architecture (EfficientNetV2-B0 + Swin-Tiny ~35.2M)
*Recommended for NVIDIA RTX 3050 (6GB VRAM) — preserves Swin-Tiny Transformer with ~41.2% parameter reduction.*

```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --checkpoint_every 100 `
  --monitor_metric macro_f1 `
  --output_dir checkpoints_swin_tiny `
  --seed 42
```

#### Memory-Optimized Swin-Tiny Variants:
- **Partial Fine-Tuning** (Freeze early stages, 21.6M trainable params, ~711MB peak VRAM):
  ```powershell
  python train_dfu_siam.py `
    --arch swin_tiny `
    --partial_finetuning `
    --enable_wagner `
    --use_hard_lmcot `
    --lambda_lmcot 0.5 `
    --lambda_mmd 0.1 `
    --lambda_orth 0.1 `
    --epochs 40 `
    --batch_size 8 `
    --output_dir checkpoints_swin_tiny_partial `
    --seed 42
  ```

- **Frozen Backbones** (Train only projections, disentangler, heads — 1.78M trainable params, ~281MB peak VRAM):
  ```powershell
  python train_dfu_siam.py `
    --arch swin_tiny `
    --freeze_backbone `
    --enable_wagner `
    --use_hard_lmcot `
    --lambda_lmcot 0.5 `
    --lambda_mmd 0.1 `
    --lambda_orth 0.1 `
    --epochs 40 `
    --batch_size 8 `
    --output_dir checkpoints_swin_tiny_frozen `
    --seed 42
  ```

---

### 2.2 Full Baseline Architecture (EfficientNetV2-S + Swin-Tiny ~59.8M)
*Original full research model.*

```powershell
python train_dfu_siam.py `
  --arch full `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --checkpoint_every 100 `
  --monitor_metric macro_f1 `
  --output_dir checkpoints_a1_full `
  --seed 42
```

---

### 2.3 Lightweight MobileViT Architecture (EfficientNetV2-B0 + MobileViT-S ~12.5M)
*Ultra-lightweight baseline.*

```powershell
python train_dfu_siam.py `
  --arch lightweight `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --checkpoint_every 100 `
  --monitor_metric macro_f1 `
  --output_dir checkpoints_lightweight `
  --seed 42
```

---

## 3. Single-Task Baselines

### 3.1 A0-Path: DFUC2021 Pathology-Only Supervised Baseline
*Wagner and multi-task auxiliary losses disabled.*

```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --loss_pathology ce `
  --lambda_path 1.0 `
  --lambda_coral 0.0 `
  --lambda_lmcot 0.0 `
  --lambda_mmd 0.0 `
  --lambda_orth 0.0 `
  --epochs 40 `
  --batch_size 8 `
  --monitor_metric macro_f1 `
  --output_dir checkpoints_a0_path `
  --seed 42
```

### 3.2 A0-Wag: Wagner-Only Ordinal Baseline
*Pathology, Siamese LMCoT, UG-MMD, and Orthogonality disabled; pure CORAL ordinal regression on deduplicated Wagner images.*

```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --lambda_path 0.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.0 `
  --lambda_mmd 0.0 `
  --lambda_orth 0.0 `
  --epochs 40 `
  --batch_size 8 `
  --monitor_metric qwk `
  --output_dir checkpoints_a0_wag `
  --seed 42
```

---

## 4. Ablation Studies (A2 – A7)

### Overview of Ablation Experiments
The ablation suite systematically isolates and removes individual components of DFU-SIAM v3 while keeping all other components constant, demonstrating the empirical contribution of each innovation:

| Experiment ID | Experiment Name | Modified Component | Purpose / Hypothesis |
| :--- | :--- | :--- | :--- |
| **A0-Path** | Pathology-Only Baseline | Wagner, LMCoT, MMD, Orth = 0 | Baseline standard classifier without multi-task transfer or metric learning |
| **A0-Wag** | Wagner-Only Baseline | Pathology, LMCoT, MMD, Orth = 0 | Pure single-task ordinal regression on Wagner images with CORAL loss |
| **A1-Full** / **A1-SwinTiny** | Full Proposed Model | All components enabled | Evaluates full synergy of dual-branch CNN+Swin, LMCoT, UG-MMD, & Orthogonality |
| **A2-NoLMCoT** | No Metric Learning | `--lambda_lmcot 0.0` | Assesses impact of Siamese Large-Margin Cotangent loss on shared representation |
| **A3-NoUGMMD** | No Domain Alignment | `--lambda_mmd 0.0` | Evaluates cross-dataset domain alignment between DFUC and Wagner domains |
| **A4-NoOrth** | No Disentanglement | `--lambda_orth 0.0` | Tests whether orthogonality ($z_{shared} \perp z_{private}$) prevents task interference |
| **A6-StandardCE** | Standard Cross-Entropy | `--loss_pathology ce` | Tests Class-Balanced Focal Loss against standard CE on long-tailed minority classes |
| **A7-NoHardMining**| Uniform LMCoT Pairs | Uniform sampling (no hard-pair mining) | Verifies effectiveness of adaptive semi-hard and hardest pair curriculum |

---

### 4.1 A2: No LMCoT Metric Learning
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.0 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --output_dir checkpoints_a2_nolmcot `
  --seed 42
```

### 4.2 A3: No Uncertainty-Gated MMD (No UG-MMD Alignment)
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.0 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --output_dir checkpoints_a3_nougmmd `
  --seed 42
```

### 4.3 A4: No Orthogonality Disentanglement Penalty
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.0 `
  --epochs 40 `
  --batch_size 8 `
  --output_dir checkpoints_a4_noorth `
  --seed 42
```

### 4.4 A6: Standard Cross-Entropy (No Class-Balanced Focal Loss)
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology ce `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --output_dir checkpoints_a6_standardce `
  --seed 42
```

### 4.5 A7: Uniform Pair Sampling (No Hard-Pair Mining in LMCoT)
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --enable_wagner `
  --loss_pathology cbfocal `
  --lambda_path 1.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --output_dir checkpoints_a7_nohardmining `
  --seed 42
```

---

## 5. Checkpoint Resumption Commands (Works for All Ablations & Baselines)

The checkpoint system works identically for **all** training runs:
- Full proposed models (A1)
- Baselines (A0-Path, A0-Wag)
- Ablation experiments (A2–A7)

Every run saves:
- `latest_checkpoint.pth` (saved every $N$ steps and immediately on `Ctrl+C`)
- `best_checkpoint.pth` (saved when validation metric improves)
- `epoch_checkpoint.pth` (saved at each epoch boundary)

### 5.1 Auto-Resume Any Run or Ablation from Its Output Directory
Just point to the respective ablation folder and append `--resume`. The script auto-detects `latest_checkpoint.pth`, restores the exact RNG, model weights, optimizer, and training batch, and seamlessly finishes the remaining epochs:

```powershell
# Example: Resuming Ablation A2 (No LMCoT)
python train_dfu_siam.py `
  --arch swin_tiny `
  --output_dir checkpoints_a2_nolmcot `
  --resume

# Example: Resuming Single-Task Wagner Baseline (A0-Wag)
python train_dfu_siam.py `
  --arch swin_tiny `
  --output_dir checkpoints_a0_wag `
  --resume
```

### 5.2 Resume from an Explicit File Path
```powershell
python train_dfu_siam.py `
  --arch swin_tiny `
  --resume_from checkpoints_a3_nougmmd/latest_checkpoint.pth
```

### 5.3 Auto-Resume in Multi-Seed Suite (`run_experiments.py`)
If you run the full automated ablation suite via `run_experiments.py` and it gets interrupted:
- Re-running the identical command automatically **skips** already finished ablations (where `best_checkpoint.pth` and `eval_val_metrics.json` exist).
- For any partially trained ablation, it detects `latest_checkpoint.pth` and **auto-resumes** from where it left off.

---

## 6. Standalone Checkpoint Evaluation

Evaluate any saved checkpoint on DFUC2021 test/val metrics, Wagner ordinal metrics (QWK, MAE, Adjacent Accuracy), and zero-shot cross-taxonomy correlation.

```powershell
python evaluate_checkpoint.py `
  --checkpoint checkpoints_swin_tiny/best_checkpoint.pth `
  --batch_size 32 `
  --output_json results_swin_tiny_best.json
```

---

## 7. Multi-Seed Automated Experiment Suite

To execute the entire ablation suite across multiple random seeds (`42`, `123`, `2026`):

```powershell
python run_experiments.py `
  --experiments A1_SwinTiny_Optimized A0_Path A0_Wag A2_NoLMCoT A3_NoUGMMD A4_NoOrth A6_StandardCE A7_NoHardMining `
  --seeds 42 123 2026 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0
```

---

## 8. Explainable AI (XAI) Inference & Faithfulness Commands

Post-hoc explainability suite operating on trained checkpoints without modifying model weights or architecture.

### 8.1 Single-Image Multi-Method Explanation
```powershell
python explain_prediction.py `
  --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth `
  --image DFUdataset/DFUC2021_train/images/305971.jpg `
  --target pathology `
  --target_id predicted `
  --xai_method all `
  --run_faithfulness `
  --output_dir outputs/xai
```

### 8.2 Explain Wagner Depth Prediction
```powershell
python explain_prediction.py `
  --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth `
  --image WagnerDataset/valid/Grade` 3/image_01.jpg `
  --target wagner `
  --target_id predicted `
  --xai_method gradcam
```

### 8.3 Batch Evaluation & Failure Mode Analysis
```powershell
# Analyze misclassified samples
python explain_dataset.py `
  --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth `
  --num_samples 20 `
  --filter incorrect_only `
  --xai_method gradcam `
  --output_dir outputs/xai_dataset_errors

# Cohort-wide faithfulness evaluation
python explain_dataset.py `
  --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth `
  --num_samples 15 `
  --filter correct_only `
  --run_faithfulness `
  --output_dir outputs/xai_dataset_cohort
```

### 8.4 Generate 5 Publication Research Figures
```powershell
python generate_xai_figures.py `
  --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth `
  --output_dir outputs/xai_figures
```

