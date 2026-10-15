# DFU-SIAM v3: Execution Commands Manual

All commands below are **PREPARED — NOT EXECUTED**.
These commands match the actual verified CLI options, datasets, scripts, and Python conda environment in the current repository.

## 0. Environment Setup (Windows PowerShell)

Before executing any commands, ensure your shell is pointed to the WagnerDS repository and conda environment `ABFormer` is active:

```powershell
# In PowerShell:
conda activate ABFormer
# Verify python points to the ABFormer environment:
Get-Command python | Select-Object -ExpandProperty Source
```

---

## 1. Unit & Numerical Stability Verification Tests

Runs the full 12-stage validation suite, including the 12 numerical boundary tests for `HardPairLMCoTLoss`:

```powershell
# [PREPARED — NOT EXECUTED] Unit & Numerical Tests
python test_all.py
```

---

## 2. Synthetic Forward & Backward Pass Smoke Test

Verifies tensor flow, loss computation, and autograd reach without touching real data:

```powershell
# [PREPARED — NOT EXECUTED] Synthetic Forward & Backward Check
python -c "
import torch
from dfu_siam_model import DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss, UncertaintyGatedMMDLoss, OrthogonalityLoss

model = DFUSiamV3().cuda()
x = torch.randn(4, 3, 224, 224, device='cuda')
z_s, z_p, p_logits = model.forward_pathology(x)
z_s_w, z_d, w_logits = model.forward_wagner(x)

cb = ClassBalancedFocalLoss().cuda()
coral = CORALLoss().cuda()
lmcot = HardPairLMCoTLoss().cuda()
mmd = UncertaintyGatedMMDLoss().cuda()
orth = OrthogonalityLoss().cuda()

l_p = cb(p_logits, torch.tensor([0, 1, 2, 3], device='cuda'))
l_c, _, _ = coral(w_logits, torch.tensor([0, 1, 2, 3], device='cuda'))
l_m = lmcot.forward_mined(z_p, torch.tensor([0, 1, 2, 3], device='cuda'), epoch=5)
l_u = mmd(z_s, z_s_w)
l_o = orth(z_s, z_p, z_d)

loss = l_p + l_c + l_m + l_u + l_o
loss.backward()
print('Synthetic Smoke Test Passed! Loss:', loss.item())
"
```

---

## 3. Short Pilot Training Execution (Sanity & Optimization Check)

Runs a 3-epoch pilot run capped at 20 batches per epoch with gradient clipping enabled:

```powershell
# [PREPARED — NOT EXECUTED] Pilot Training
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --pilot `
  --epochs 3 `
  --max_batches_per_epoch 20 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_pilot `
  --seed 42
```

---

## 4. Baselines (Pathology-Only & Wagner-Only)

### A0-Path: Baseline Pathology Model
Trains the pathology model with standard Cross-Entropy, without metric learning, MMD alignment, or disentanglement losses:

```powershell
# [PREPARED — NOT EXECUTED] Baseline Pathology Run (A0-Path)
python train_dfu_siam.py `
  --loss_pathology ce `
  --lambda_coral 0.0 `
  --lambda_lmcot 0.0 `
  --lambda_mmd 0.0 `
  --lambda_orth 0.0 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_baseline_pathology `
  --seed 42
```

### A0-Wag: Baseline Wagner Ordinal Model
Trains the model on Wagner depth classification only with CORAL ordinal loss, without multi-task cross-cohort alignment:

```powershell
# [PREPARED — NOT EXECUTED] Baseline Wagner Run (A0-Wag)
python train_dfu_siam.py `
  --enable_wagner `
  --lambda_path 0.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.0 `
  --lambda_mmd 0.0 `
  --lambda_orth 0.0 `
  --monitor_metric qwk `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_baseline_wagner `
  --seed 42
```

---

## 5. Full Proposed Model Training (Experiment A1 — DFU-SIAM v3)

Runs full multi-task, task-disentangled training with all losses active ($\lambda_{\text{lmcot}} = 0.5$):

```powershell
# [PREPARED — NOT EXECUTED] Full Proposed DFU-SIAM v3 (A1)
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_v3_full `
  --seed 42
```

---

## 6. Ablation Experiments

### A2 — No LMCoT Ablation ($\lambda_{\text{lmcot}} = 0.0$)
```powershell
# [PREPARED — NOT EXECUTED] Ablation A2: No LMCoT
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.0 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_ablation_no_lmcot `
  --seed 42
```

### A3 — No Uncertainty-Gated MMD Ablation ($\lambda_{\text{mmd}} = 0.0$)
```powershell
# [PREPARED — NOT EXECUTED] Ablation A3: No UG-MMD
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.0 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_ablation_no_mmd `
  --seed 42
```

### A4 — No Orthogonality Ablation ($\lambda_{\text{orth}} = 0.0$)
```powershell
# [PREPARED — NOT EXECUTED] Ablation A4: No Orthogonality
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.0 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_ablation_no_orth `
  --seed 42
```

### A6 — Standard Cross-Entropy Comparison (in place of CB-Focal)
```powershell
# [PREPARED — NOT EXECUTED] Ablation A6: Standard CE
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology ce `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_ablation_ce `
  --seed 42
```

### A7 — Uniform LMCoT Pairs (No Hard Mining)
```powershell
# [PREPARED — NOT EXECUTED] Ablation A7: No Hard Mining
python train_dfu_siam.py `
  --enable_wagner `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_ablation_no_hard_mining `
  --seed 42
```

---

## 7. Multi-Seed Replication Protocol

Execute Full v3 across 3 independent random seeds with identical model hyperparameters:

```powershell
# [PREPARED — NOT EXECUTED] Seed 42 (Full v3)
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_v3_seed42 `
  --seed 42

# [PREPARED — NOT EXECUTED] Seed 123 (Full v3)
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_v3_seed123 `
  --seed 123

# [PREPARED — NOT EXECUTED] Seed 2026 (Full v3)
python train_dfu_siam.py `
  --enable_wagner `
  --use_hard_lmcot `
  --loss_pathology cbfocal `
  --focal_beta 0.999 `
  --focal_gamma 2.0 `
  --lambda_coral 1.0 `
  --lambda_lmcot 0.5 `
  --lambda_mmd 0.1 `
  --lambda_orth 0.1 `
  --epochs 40 `
  --batch_size 8 `
  --grad_clip 1.0 `
  --output_dir checkpoints_v3_seed2026 `
  --seed 2026
```

---

## 8. Checkpoint Evaluation & Output Export

Evaluate trained checkpoints and export performance to JSON:

```powershell
# [PREPARED — NOT EXECUTED] Evaluation for Seed 42
python evaluate_checkpoint.py `
  --checkpoint checkpoints_v3_seed42/dfu_siam_v3_best.pth `
  --output_json results_v3_seed42.json

# [PREPARED — NOT EXECUTED] Evaluation for Seed 123
python evaluate_checkpoint.py `
  --checkpoint checkpoints_v3_seed123/dfu_siam_v3_best.pth `
  --output_json results_v3_seed123.json

# [PREPARED — NOT EXECUTED] Evaluation for Seed 2026
python evaluate_checkpoint.py `
  --checkpoint checkpoints_v3_seed2026/dfu_siam_v3_best.pth `
  --output_json results_v3_seed2026.json
```

---

## 9. Multi-Seed Results Aggregation

Computes mean and standard deviation across seeds:

```powershell
# [PREPARED — NOT EXECUTED] Multi-Seed Aggregation
python aggregate_seeds.py `
  --inputs results_v3_seed42.json results_v3_seed123.json results_v3_seed2026.json `
  --output aggregated_v3_results.json
```

---

## 10. Figures & Publication Plots Generation

Generates high-resolution confusion matrix heatmaps and per-class performance figures:

```powershell
# [PREPARED — NOT EXECUTED] Report Figure Generation
python plot_experiment_reports.py `
  --eval_json results_v3_seed42.json `
  --output_dir publication_figures
```
