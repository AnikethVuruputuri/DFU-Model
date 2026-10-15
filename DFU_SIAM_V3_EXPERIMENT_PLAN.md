# DFU-SIAM v3: Scientific Experiment & Evaluation Plan

## 1. Executive Summary & Objective

**DFU-SIAM v3 (Diabetic Foot Ulcer – Shared Intelligence and Alignment Model, Version 3)** is a dual-backbone, multi-task representation learning architecture designed for automated diabetic foot ulcer characterization:
1. **Long-Tailed Pathology Classification (DFUC2021 Cohort)**: 4-class classification (`none`, `infection`, `ischaemia`, `both`).
2. **Wagner Depth/Severity Grading (Wagner Dataset Cohort)**: 5-class ordinal grading (Grades 0 to 4).

The primary research objective is to learn a clean, task-disentangled feature representation decomposing fused 2560-D hybrid features (EfficientNetV2-S + Swin-Tiny) into:
- $z_{\text{shared}} \in \mathbb{R}^{768}$: Transferable, clinical wound features common across datasets.
- $z_{\text{path\_private}} \in \mathbb{R}^{512}$: Cohort-specific pathology features strictly isolated from depth.
- $z_{\text{depth\_private}} \in \mathbb{R}^{512}$: Cohort-specific depth features strictly isolated from infection/ischaemia.

---

## 2. Model Architecture & Tensor Flow

```
Input Image: x ∈ ℝ^(B × 3 × 224 × 224)
  ├── Branch 1: EfficientNetV2-S ──> Linear(1280, 1280) + GELU ──> ℝ^(B × 1280)
  └── Branch 2: Swin-Tiny (ViT)  ──> Linear(768, 1280)  + GELU ──> ℝ^(B × 1280)
Fused Feature Representation: Concatenation [Branch 1; Branch 2] ──> ℝ^(B × 2560)
  ├── Shared Projector:  Linear(2560, 1024) ──> BN ──> GELU ──> Linear(1024, 768) ──> BN ──> L2Norm ──> z_shared ∈ ℝ^(B × 768)
  ├── Path Projector:    Linear(2560, 768)  ──> BN ──> GELU ──> Linear(768, 512)  ──> BN ──> L2Norm ──> z_path_private ∈ ℝ^(B × 512)
  └── Depth Projector:   Linear(2560, 768)  ──> BN ──> GELU ──> Linear(768, 512)  ──> BN ──> L2Norm ──> z_depth_private ∈ ℝ^(B × 512)

Task Inferences:
  Pathology Head: [z_shared; z_path_private] (1280-D) ──> Linear(1280, 512) ──> BN ──> ReLU ──> Linear(512, 4) ──> 4 Logits
  Wagner Head:    [z_shared; z_depth_private] (1280-D) ──> Linear(1280, 512) ──> BN ──> ReLU ──> Linear(512, 4) ──> 4 CORAL Cutoffs
```

---

## 3. Loss Functions & Optimization Formulation

$$\mathcal{L}_{\text{total}} = \lambda_{\text{path}} \mathcal{L}_{\text{path\_CB}} + \lambda_{\text{wagner}} \mathcal{L}_{\text{wagner\_CORAL}} + \lambda_{\text{metric}} \mathcal{L}_{\text{Hard-LMCoT}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{UG-MMD}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$

### Default Loss Weighting Schedule
- $\lambda_{\text{path}} = 1.0$ (Class-Balanced Focal Loss, $\beta = 0.999$, $\gamma = 2.0$)
- $\lambda_{\text{wagner}} = 1.0$ (CORAL Cumulative Ordinal Binary Cross-Entropy)
- $\lambda_{\text{metric}} = 0.5$ (Hard-Pair LMCoT Metric Learning, Margin $m=2.0$, $d_{\max}=10.0$, Warmup=5 epochs — scaled from 1.0 to 0.5 based on empirical batch loss magnitude profiling: mean=1.36 vs CB-Focal=0.80)
- $\lambda_{\text{mmd}} = 0.1$ (Uncertainty-Gated MMD on $z_{\text{shared}}$ with detached confidence weights)
- $\lambda_{\text{orth}} = 0.1$ (Disentanglement Regularization: $z_{\text{shared}} \perp z_{\text{path\_private}}$ and $z_{\text{shared}} \perp z_{\text{depth\_private}}$)

### Optimizer & Schedulers
- Optimizer: `AdamW` (Separate parameter groups: Backbone $\text{LR}=3 \times 10^{-6}$, Projection/Heads $\text{LR}=1 \times 10^{-4}$, weight decay $10^{-3}$)
- Scheduler: `CosineAnnealingLR` ($T_{\max}=40$, $\eta_{\min}=10^{-7}$)
- Gradient Scaler: AMP FP16 with `init_scale=1024.0` (unscaled before clipping)
- Gradient Clipping: `clip_grad_norm_` with threshold $= 1.0$

---

## 4. Controlled Experimental Matrix

| ID | Name | Components Enabled | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **A0-Path** | **Pathology Baseline** | Standard Cross-Entropy on DFUC; no Wagner, no metric, no MMD, no orth | Quantifies the performance floor for pathology classification using ordinary supervised learning. |
| **A0-Wag** | **Wagner Ordinal Baseline** | Wagner CORAL head trained on Wagner dataset only | Isolates single-task Wagner depth grading to verify if multi-task alignment aids or hurts depth prediction. |
| **A1** | **Full DFU-SIAM v3** | CB-Focal + CORAL + Hard-LMCoT ($\lambda=0.5$) + UG-MMD + Orthogonality | Primary proposed method with all disentangled and multi-task components enabled. |
| **A2** | **No LMCoT Ablation** | Full v3 with $\lambda_{\text{lmcot}} = 0.0$ | Evaluates the specific contribution of tangent metric learning on minority class discrimination. |
| **A3** | **No UG-MMD Ablation** | Full v3 with $\lambda_{\text{mmd}} = 0.0$ | Evaluates whether domain distribution alignment on $z_{\text{shared}}$ prevents negative cross-cohort transfer. |
| **A4** | **No Orthogonality Ablation** | Full v3 with $\lambda_{\text{orth}} = 0.0$ | Tests whether enforcing mathematical subspace independence ($z_{\text{shared}} \perp z_{\text{private}}$) prevents representation collapse. |
| **A5** | **No Task Disentanglement Analysis** | Documented structural ablation analysis (see Section 4.1 below) | Analyzes whether decomposing representations into private/shared subspaces outperforms monolithic concatenation. |
| **A6** | **Standard Focal Comparison** | Full v3 replacing CB-Focal with standard Cross-Entropy (`--loss_pathology ce`) | Demonstrates the exact effect of effective-number class weighting ($E_c = 1 - \beta^{n_c}$). |
| **A7** | **No Hard Mining Ablation** | Full v3 with uniform/random LMCoT pairs (`--use_hard_lmcot` flag omitted) | Evaluates whether active boundary mining (None $\leftrightarrow$ Infection, Ischaemia $\leftrightarrow$ Both) outperforms uniform sampling. |

### 4.1 Scientific Evaluation of Ablation A5 (Task Disentanglement)
In the DFU-SIAM v3 architecture, task disentanglement is parameterized by three parallel sub-projectors in `TaskDisentangler`: `shared_proj` (768-D), `path_proj` (512-D), and `depth_proj` (512-D), which directly supply the 1280-D inputs to `classifier` and `wagner_head`.
- **Ablation A4 ($\lambda_{\text{orth}} = 0.0$)** evaluates the system when the geometric penalty enforcing linear independence ($z_{\text{shared}} \perp z_{\text{private}}$) is removed, allowing features to freely entangle.
- A monolithic single-vector ablation (e.g. projecting 2560-D into a single 1280-D vector shared identically across both heads without separate private channels) requires redefining tensor input dimensions across classification heads and forward signatures. Therefore, **A4 represents the direct, clean, controlled in-architecture ablation of disentanglement without altering network topology or parameter count**.

### 4.2 Wagner Dataset Deduplication & Single-Task Baseline Protocol
1. **Wagner Deduplication**:
   - An exact MD5 hash audit identified 37 Grade 0 images in `WagnerDataset/train` that were byte-for-byte identical to images in `valid` (21 images) and `test` (16 images), alongside 111 augmented variants derived from those base images in `train`.
   - All 148 duplicate/augmented images are tracked in [`wagner_dedup_manifest.json`](file:///c:/Users/srian/Desktop/WagnerDS/wagner_dedup_manifest.json) and excluded at loader instantiation by `WagnerSingleDataset`.
   - **Post-Deduplication Counts**: Train: **11,491** (Grade 0: 1,852, Grade 1: 2,240, Grade 2: 2,345, Grade 3: 2,744, Grade 4: 2,310). Valid: **303** intact. Test: **157** intact. Zero hash leakage remains.
   - *Patient Metadata Note*: Because `WagnerDataset` contains no patient IDs or clinical records, patient-level separation cannot be claimed.
2. **A0-Wag (Wagner-Only Baseline) Formulation**:
   - Executed via `--enable_wagner --lambda_path 0.0 --lambda_coral 1.0 --lambda_lmcot 0.0 --lambda_mmd 0.0 --lambda_orth 0.0 --monitor_metric qwk`.
   - Operates in a dedicated single-task loop over `wagner_train_loader`, computing only $\mathcal{L}_{\text{coral}}$ with zero forward passes or gradients through the pathology classification head.
   - Checkpoint selection monitors Wagner QWK (`--monitor_metric qwk`) or MAE (`--monitor_metric mae`).

---

## 5. Evaluation Protocol & Publication Metrics

All evaluations must be performed using the verified standalone evaluator [`evaluate_checkpoint.py`](file:///c:/Users/srian/Desktop/WagnerDS/evaluate_checkpoint.py) on identical held-out validation sets:
- **Pathology Evaluation (DFUC2021)**:
  - Macro-F1 (primary competition benchmark metric)
  - Accuracy
  - Macro-Precision & Macro-Recall (Unweighted Average Recall / UAR)
  - Per-class F1 for `none`, `infection`, `ischaemia`, `both`
  - Matthews Correlation Coefficient (MCC)
  - Macro-AUC (One-vs-Rest)
  - Multiclass Confusion Matrix
- **Wagner Depth Evaluation (Wagner Dataset)**:
  - Accuracy
  - Mean Absolute Error (MAE in ordinal grade units)
  - Adjacent Accuracy ($\pm 1$ grade tolerance)
  - Macro-F1
  - Quadratic Weighted Kappa (QWK)
  - 5×5 Ordinal Confusion Matrix
- **Cross-Taxonomy Consistency**:
  - Spearman rank correlation ($\rho$)
  - Pearson correlation ($r$)
  - Exploratory 16-cell University of Texas (UT) wound matrix (clearly documented as an exploratory probabilistic visualization, NOT ground truth)

---

## 6. Multi-Seed Reproducibility Protocol

Each core configuration (**A0 Baseline**, **A1 Full v3**, and key ablations **A2, A3, A4**) must be run across three independent random seeds:
- Seed 42
- Seed 123
- Seed 2026

Runs must be stored in independent directories (`checkpoints_v3_seed42`, etc.). Aggregation across seeds is performed via [`aggregate_seeds.py`](file:///c:/Users/srian/Desktop/WagnerDS/aggregate_seeds.py), reporting exact $\text{mean} \pm \text{std}$ across metrics without pooling confusion matrices before metric calculation.
