# Pending Experiments and Scientific Execution Tracker: DFU-WAGNER net

**Document Purpose**: Provides a rigorous checklist and command reference for all experimental evaluations required to populate the empirical placeholders in the **DFU-WAGNER net** research manuscript.

---

## 1. Research Integrity & Anti-Fabrication Commitment

> [!IMPORTANT]
> In strict accordance with scientific integrity guidelines:
> - **ZERO fabricated, simulated, or estimated performance numbers** are permitted in the manuscript.
> - All reported metrics must originate from verifiable evaluation logs, confusion matrices, and checkpoint files saved by `evaluate_checkpoint.py` or `aggregate_seeds.py`.
> - Validation metrics must be clearly distinguished from test metrics.
> - Where external test set labels are held out by challenge organizers (DFUC2021 test set), evaluation is conducted on the benchmark-standard stratified held-out validation split ($n = 1,191$), explicitly declared as such.

---

## 2. Experimental Checklist & Execution Status

| Experiment ID | Configuration Name | Status | Output Directory / Log | Monitored Metric | Target Table / Figure in Manuscript |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **A1-Primary** | **Full Proposed DFU-WAGNER net (Seed 42)** | **Training / Evaluation Pipeline** | `checkpoints_a1_full_balanced/` | Joint / Macro-F1 | Table IV, Table VI, Table VII, Fig 4, 5, 6 |
| **A0-Path** | Pathology-Only Supervised Baseline | Pending Execution | `checkpoints_a0_path/` | Macro-F1 | Table V (Ablation) |
| **A0-Wag** | Wagner-Only Ordinal Baseline | Pilot Verified / Full Run | `checkpoints_a0_wagner/` | QWK / MAE | Table V (Ablation), Table VII |
| **A2** | No LMCoT Metric Learning ($\lambda_{\text{lmcot}}=0$) | Pending Execution | `checkpoints_a2_no_lmcot/` | Macro-F1 | Table V (Ablation) |
| **A3** | No UG-MMD Alignment ($\lambda_{\text{mmd}}=0$) | Pending Execution | `checkpoints_a3_no_mmd/` | Macro-F1 | Table V (Ablation) |
| **A4** | No Orthogonality Regularization ($\lambda_{\text{orth}}=0$) | Pending Execution | `checkpoints_a4_no_orth/` | Macro-F1 | Table V (Ablation) |
| **A6** | Standard CE instead of CB-Focal | Pending Execution | `checkpoints_a6_ce/` | Ischaemia F1 | Table V (Ablation) |
| **A7** | Uniform LMCoT Pairs (No Hard Mining) | Pending Execution | `checkpoints_a7_uniform_lmcot/` | Macro-F1 | Table V (Ablation) |
| **Seed Runs** | Multi-Seed Reproducibility (Seeds 123, 2026) | Pending Execution | `checkpoints_v3_seed123/`, `...2026/` | Mean $\pm$ Std | Table IV (Final Confidence Intervals) |
| **XAI-GradCAM**| Multi-Branch Grad-CAM Visualizations | Pending Generation | `output_cam_v3/` | Qualitative Attribution | Figure 7 |

---

## 3. Standardized Execution Commands

All commands run under the active Conda/Python environment with CUDA acceleration:

### 3.1 Primary Candidate Evaluation (A1)
To generate the definitive evaluation metrics and confusion matrices for the trained A1 checkpoint:
```bash
python evaluate_checkpoint.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --arch full \
    --data_dir DFUdataset \
    --wagner_dir WagnerDataset \
    --batch_size 16 \
    --output_dir output_eval_a1_full
```

### 3.2 Single-Task Baselines
#### A0-Path (Pathology-Only Baseline):
```bash
python train_dfu_siam.py \
    --arch full \
    --lambda_path 1.0 \
    --loss_pathology ce \
    --lambda_coral 0.0 \
    --lambda_lmcot 0.0 \
    --lambda_mmd 0.0 \
    --lambda_orth 0.0 \
    --epochs 40 \
    --batch_size 8 \
    --lr_backbone 1e-5 \
    --lr_head 2e-4 \
    --output_dir checkpoints_a0_path
```

#### A0-Wag (Wagner-Only Baseline):
```bash
python train_dfu_siam.py \
    --arch full \
    --enable_wagner \
    --lambda_path 0.0 \
    --lambda_coral 1.0 \
    --lambda_lmcot 0.0 \
    --lambda_mmd 0.0 \
    --lambda_orth 0.0 \
    --monitor_metric qwk \
    --epochs 40 \
    --batch_size 8 \
    --lr_backbone 1e-5 \
    --lr_head 2e-4 \
    --output_dir checkpoints_a0_wagner
```

### 3.3 Core Ablation Runs
#### A2 (Without LMCoT Metric Learning):
```bash
python train_dfu_siam.py \
    --arch full \
    --enable_wagner \
    --lambda_path 1.0 \
    --lambda_coral 1.0 \
    --lambda_lmcot 0.0 \
    --lambda_mmd 0.1 \
    --lambda_orth 0.1 \
    --epochs 40 \
    --output_dir checkpoints_a2_no_lmcot
```

#### A3 (Without UG-MMD Domain Alignment):
```bash
python train_dfu_siam.py \
    --arch full \
    --enable_wagner \
    --lambda_path 1.0 \
    --lambda_coral 1.0 \
    --lambda_lmcot 0.5 \
    --use_hard_lmcot \
    --lambda_mmd 0.0 \
    --lambda_orth 0.1 \
    --epochs 40 \
    --output_dir checkpoints_a3_no_mmd
```

#### A4 (Without Orthogonality Regularization):
```bash
python train_dfu_siam.py \
    --arch full \
    --enable_wagner \
    --lambda_path 1.0 \
    --lambda_coral 1.0 \
    --lambda_lmcot 0.5 \
    --use_hard_lmcot \
    --lambda_mmd 0.1 \
    --lambda_orth 0.0 \
    --epochs 40 \
    --output_dir checkpoints_a4_no_orth
```

---

## 4. Procedure for Slotting Empirical Results into Manuscript

Once an evaluation run finishes and produces its summary JSON report:
1. Locate the placeholder tag:
   - `[FINAL TEST RESULTS TO BE INSERTED AFTER EXPERIMENTS]` in Abstract.
   - `[FINAL TEST RESULTS PENDING]` in Table IV and Section 6.1.
   - `[CONFUSION MATRIX AND PER-CLASS METRICS PENDING]` in Table VI and Figure 5.
   - `[FINAL WAGNER TEST RESULTS PENDING]` in Table VII and Figure 6.
   - `[ABLATION RESULTS PENDING]` in Table V and Section 6.4.
2. Read the exact numerical fields from the evaluation output:
   - `macro_f1` $\to$ Insert into Abstract, Table IV, Table V, Table VI.
   - `ischaemia_f1` $\to$ Insert into Table IV, Table V, Table VI.
   - `qwk`, `mae`, `adjacent_acc` $\to$ Insert into Table V, Table VII.
3. Replace the placeholder tag with the verified numerical value (reporting exact percentage or 4 decimal places).
4. In Section 7 (Discussion) and Section 11 (Conclusion), replace provisional qualitative sentences with precise comparative statements based directly on the inserted metrics.
