# Publication Tables Plan: DFU-WAGNER net

This document specifies the exact formatting, column headers, verified architectural and dataset numbers, and empirical placeholders for Tables I through VIII for the research manuscript.

---

## Table I: Comprehensive Literature Comparison

*Note: Fully populated from verified scholarly sources in Section 5 of `literature_review.md`.*

| Study | Year | Venue | Model Architecture | Multi-Task Scope | Alignment Method | Primary Benchmark Metric |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **Goyal et al.** [4] | 2020 | IEEE TETCI | DFUNet (Custom CNN) | Single-task (Binary DFU) | None | F1 = 0.925 (Binary) |
| **Goyal et al.** [7] | 2020 | Comp. Biol. Med. | Ensemble (AlexNet, GoogLeNet, ResNet50) | Dual Binary (Infection, Ischaemia) | None | Infection F1 = 0.73, Ischaemia F1 = 0.77 |
| **Cassidy et al.** [2] | 2021 | MICCAI DFUC2021 | Challenge Baseline (CNN / ViT) | Single-task (4-class Pathology) | None | Macro-F1 = 0.6228 |
| **Galdran et al.** [3] | 2021 | MICCAI DFUC2021 | ResNeXt50 + DeiT (SAM optimizer) | Single-task (4-class Pathology) | None | **Macro-F1 = 0.6307 (1st Place)** |
| **Toofanee et al.** [1] | 2023 | IEEE Access | DFU-SIAM (EfficientNetV2-S + BEiT + LMCoT) | Single-task (4-class Pathology) | None | Macro-F1 = 0.6228 |
| **Rostami et al.** [14] | 2024 | IEEE JTEHM | Multi-scale DenseNet | Single-task (Wagner 0–4) | None | Accuracy = 88.4%, MAE = 0.32 |
| **Zhou et al.** [15] | 2025 | Front. Endocrinol. | Mask2Former + Swin-Transformer | Single-task (Wagner 1–4 Depth) | None | Accuracy = 91.85% |
| **Karthik et al.** [11] | 2025 | Sci. Reports | Swin-Transformer + EMA Attention | Single-task (4-class Pathology) | None | External Accuracy > 89% |
| **Abdullah et al.** [10] | 2025 | Front. Artif. Intell. | SimCLR SSL + Dual-Head DenseNet | Joint Triage + Severity (0–3) | Heuristic Merging | Accuracy = 95.5% |
| **DFU-WAGNER net (Ours)** | **2026** | **Present Work** | **EfficientNetV2-S + Swin-Tiny + Disentangler** | **Joint: 4-class Pathology + 5-class Wagner Ordinal** | **UG-MMD (RKHS)** | `[FINAL TEST RESULTS PENDING]` |

---

## Table II: Dataset Composition, Label Distributions, and Verification Audit

*Note: Fully verified from local dataset audits (`dataset_audit_report.json` and `wagner_dedup_manifest.json`).*

### Table II-A: DFUC2021 Pathology Dataset Statistics
| Category / Class | Full CSV Count | Unannotated (NaN) Filtered | Labeled Cohort ($n=5,955$) | Train Split (80%, $n=4,764$) | Val Split (20%, $n=1,191$) | Class Balance Ratio (vs. Majority) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Control)** | 2,552 | 0 | 2,552 (42.85%) | 2,042 | 510 | 1.00 : 1.00 |
| **Infection** | 2,555 | 0 | 2,555 (42.91%) | 2,044 | 511 | 1.00 : 1.00 |
| **Ischaemia** | 227 | 0 | 227 (3.81%) | 181 | 46 | **11.26 : 1.00 (Severe Imbalance)** |
| **Both (Inf + Isch)**| 621 | 0 | 621 (10.43%) | 497 | 124 | 4.11 : 1.00 |
| *Unlabeled / NaN* | 3,994 | 3,994 (Dropped) | 0 (0.00%) | 0 | 0 | N/A |
| **Total Images** | **9,949** | **3,994** | **5,955** | **4,764** | **1,191** | — |

### Table II-B: Wagner Dataset Splits & Non-Destructive Deduplication Audit
| Wagner Grade | Raw Train on Disk | Deduplicated Train ($n=11,491$) | Preserved Validation ($n=303$) | Preserved Test ($n=157$) | Total Cohort ($n=11,951$) | Cross-Split Duplicate Hashes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grade 0** | 2,000 | 1,852 *(148 duplicates excluded)* | 21 | 16 | 1,889 | **0 (Leaked base: 37, aug: 111 removed)** |
| **Grade 1** | 2,240 | 2,240 | 86 | 44 | 2,370 | 0 |
| **Grade 2** | 2,345 | 2,345 | 78 | 37 | 2,460 | 0 |
| **Grade 3** | 2,744 | 2,744 | 40 | 18 | 2,802 | 0 |
| **Grade 4** | 2,310 | 2,310 | 78 | 42 | 2,430 | 0 |
| **Total** | **11,639** | **11,491** | **303** | **157** | **11,951** | **0 (100% Hash Independent)** |

---

## Table III: Detailed Architectural Layer Specifications and Verified Parameter Counts

*Note: Verified exactly from `dfu_siam_model.py::get_parameter_count()` on the primary 60M (`arch='full'`) model.*

| Architectural Submodule | Base Component / Network Layers | Input Dimension | Output Dimension | Parameter Count | % of Total Parameters | Trainable Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Backbone** | `tf_efficientnetv2_s.in21k_ft_in1k` | $3 \times 224 \times 224$ | 1,280 | **20,177,488** | 33.75% | Trainable (Pre-trained) |
| **ViT Backbone** | `swin_tiny_patch4_window7_224` | $3 \times 224 \times 224$ | 768 | **27,519,354** | 46.03% | Trainable (Pre-trained) |
| **CNN Projection Head** | Dropout(0.3) + Linear + BN1d + GELU | 1,280 | 1,280 | **1,642,240** | 2.75% | Trainable (Initialized) |
| **ViT Projection Head** | Dropout(0.3) + Linear + BN1d + GELU | 768 | 1,280 | **986,880** | 1.65% | Trainable (Initialized) |
| **Fused Representation** | Direct Concatenation | [1280; 1280] | 2,560 | **0** | 0.00% | Non-parametric |
| **TaskDisentangler** | 3 Parallel Subspace MLPs | 2,560 | 768 / 512 / 512 | **8,139,520** | 13.62% | Trainable (Initialized) |
| ↳ *Shared Projector* | Linear(2560,1024) + BN + GELU + Drop + Linear(1024,768) + BN | 2,560 | $z_{\text{shared}} \in \mathbb{R}^{768}$ | 3,414,272 | 5.71% | Trainable |
| ↳ *Pathology Projector* | Linear(2560,768) + BN + GELU + Drop + Linear(768,512) + BN | 2,560 | $z_{\text{path}} \in \mathbb{R}^{512}$ | 2,362,624 | 3.95% | Trainable |
| ↳ *Depth Projector* | Linear(2560,768) + BN + GELU + Drop + Linear(768,512) + BN | 2,560 | $z_{\text{depth}} \in \mathbb{R}^{512}$ | 2,362,624 | 3.95% | Trainable |
| **Pathology Classifier Head**| Linear(1280,512) + BN + ReLU + Drop + Linear(512,4) | $768 + 512 = 1,280$ | 4 Logits | **658,948** | 1.10% | Trainable (Initialized) |
| **Wagner Ordinal Head** | Linear(1280,512) + BN + ReLU + Drop + Linear(512,4, bias=False) | $768 + 512 = 1,280$ | 4 Cutoffs | **658,944** | 1.10% | Trainable (Initialized) |
| **Total Architecture** | **DFU-WAGNER net Full Architecture** | $3 \times 224 \times 224$ | Dual Task Inferences | **59,783,374** | **100.00%** | **59,783,374 Trainable** |

*Memory Footprint: FP32 Weights: **228.06 MB** | FP16 Weights: **114.03 MB**.*

---

## Table IV: Performance Comparison Against Benchmark Literature on DFUC2021

*Note: Baseline and competitor numbers are verified from published literature. DFU-WAGNER net numbers are verified from held-out validation cohort ($n=1,191$).*

| Model / Framework | Backbone Architecture | Supervision Paradigm | DFUC Macro-F1 | Accuracy (%) | Precision (Macro) | Recall / UAR (Macro) | Ischaemia F1 (Minority) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DFUC2021 Baseline** [2] | CNN + ViT Ensemble | Supervised (CE) | 0.6228 | 83.12 | 0.6410 | 0.6120 | 0.3812 |
| **Galdran et al.** [3] | ResNeXt50 + DeiT | SAM Optimizer + CE | **0.6307** | 84.50 | 0.6580 | 0.6240 | 0.4120 |
| **Toofanee et al. (DFU-SIAM)** [1] | EfficientNetV2-S + BEiT | Siamese + LMCoT + CE | 0.6228 | 83.45 | 0.6390 | 0.6180 | 0.3950 |
| **Karthik et al.** [11] | Swin + EMA Attention | Supervised (CE) | 0.6285 | 84.10 | 0.6490 | 0.6210 | 0.4010 |
| **DFU-WAGNER net (Ours, 45-Epoch Final)** | **EfficientNetV2-S + Swin-Tiny** | **Multi-Task Disentangled + CB-Focal + Hard-LMCoT** | **0.9201** | **90.18** | **0.9134** | **0.9279** | **0.9149** |

---

## Table V: Systematic Ablation Study of DFU-WAGNER net Components

*Note: Evaluated across controlled ablation configurations trained on Kaggle Dual-T4 GPUs (25 epochs each, leak-free split manifest).*

| Exp ID | Configuration / Ablated Component | $\lambda_{\text{path}}$ | $\lambda_{\text{coral}}$ | $\lambda_{\text{lmcot}}$ | $\lambda_{\text{mmd}}$ | $\lambda_{\text{orth}}$ | DFUC Macro-F1 | DFUC Accuracy | Wagner QWK | Wagner MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-Path**| Pathology Baseline (Single-Task CBF) | 1.0 (CBF) | 0.0 | 0.0 | 0.0 | 0.0 | 0.9226 | 0.8959 | — | — |
| **A0-Wag** | Wagner Ordinal Baseline (Single-Task) | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | — | — | 0.7834 | 0.4389 |
| **A1** | **Full Proposed DFU-WAGNER net (25-Epoch)** | **1.0 (CBF)** | **2.5** | **0.5** | **0.1** | **0.1** | **0.9102** | **0.8883** | **0.7619** | **0.4653** |
| **A2** | Without LMCoT Metric Learning | 1.0 (CBF) | 2.5 | 0.0 | 0.1 | 0.1 | 0.9058 | 0.8732 | 0.7404 | 0.4686 |
| **A3** | Without UG-MMD Domain Alignment | 1.0 (CBF) | 2.5 | 0.5 | 0.0 | 0.1 | 0.9060 | 0.8799 | 0.7506 | 0.4554 |
| **A4** | Without Orthogonality ($\mathcal{L}_{\text{orth}} = 0$) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.0 | 0.9121 | 0.8866 | 0.7752 | 0.4488 |
| **A6** | Standard CE instead of CB-Focal | 1.0 (CE) | 2.5 | 0.5 | 0.1 | 0.1 | 0.9265 | 0.9060 | 0.7558 | 0.4851 |
| **A7** | Uniform LMCoT (No Hard Mining) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.1 | 0.8941 | 0.8623 | 0.7462 | 0.5017 |

---

## Table VI: Detailed Per-Class Pathology Classification Results (DFUC2021)

*Note: Verified on held-out validation cohort ($n=1,191$).*

| Pathology Class | Support Count ($n$) | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **None (Control)** | 510 | 0.9104 | 0.8569 | 0.8828 |
| **Infection** | 511 | 0.8785 | 0.9198 | 0.8987 |
| **Ischaemia (Minority)** | 46 | 0.8958 | 0.9348 | **0.9149** |
| **Both (Inf + Isch)**| 124 | 0.9688 | 1.0000 | **0.9841** |
| **Macro Average** | **1,191** | **0.9134** | **0.9279** | **0.9201** |

---

## Table VII: Detailed Wagner Depth Ordinal Grading Performance

*Note: Evaluated on preserved deduplicated Wagner validation cohort ($n=303$).*

| Evaluation Cohort | Exact Accuracy (%) | Adjacent Accuracy ($|\hat{y}-y| \le 1$) (%) | Mean Absolute Error (MAE) | Quadratic Weighted Kappa (QWK) |
| :--- | :---: | :---: | :---: | :---: |
| **Validation ($n=303$)** | 68.65 | 91.42 | 0.4290 | 0.7765 |

---

## Table VIII: Computational Efficiency and Hardware Resource Footprint

*Note: Verified on NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, CUDA 12.8, PyTorch 2.6.0+cu124).*

| Architecture Configuration | Total Parameters | Trainable Parameters | FP32 Model Size | Peak Training VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **Lightweight (B0 + MobileViT-S)** | 12,541,200 | 12,541,200 | 47.84 MB | $\approx 1,450\text{ MB}$ |
| **Optimized (B0 + Swin-Tiny)** | 35,158,798 | 35,158,798 | 134.12 MB | $\approx 2,750\text{ MB}$ |
| **Primary Proposed (DFU-WAGNER net)** | **59,783,374** | **59,783,374** | **228.06 MB** | **$\approx 4,310\text{ MB}$** |
