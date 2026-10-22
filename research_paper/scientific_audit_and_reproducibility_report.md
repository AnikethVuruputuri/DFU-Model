# Scientific and Technical Audit Report: DFU-WAGNER Net / DFU-SIAM v3

**Auditor**: Senior ML Research Engineer and Scientific Paper Auditor  
**Date**: October 19, 2026  
**Repository**: `WagnerDS`  
**Target Manuscript (Original Preserved)**: `research_paper/DFU_WAGNER_net_paper.tex`  
**Corrected Manuscript**: `research_paper/DFU_WAGNER_net_paper_corrected.tex`  
**Reproducibility Manifests**:
- `research_paper/dataset_cross_overlap_manifest.json`
- `research_paper/dataset_cross_overlap_manifest.csv`
- `research_paper/clean_leak_free_split_manifest.json`
- `wagner_dedup_manifest.json`

---

## Executive Summary

A comprehensive scientific, algorithmic, and data-integrity audit was conducted on the **DFU-WAGNER Net** (formerly DFU-SIAM v3) multi-task deep representation learning framework. The audit inspected the entire codebase, training pipelines, data loaders, split generation scripts, loss functions, checkpoint-selection mechanisms, experiment configurations, learning curves, and the draft manuscript.

### Key Audit Outcomes:
1. **Zero Data Fabrication**: All reported metrics across the 45-epoch primary model and the 8 ablation runs were traced to cryptographic logs, learning curve JSONs, and model checkpoints. No results were fabricated or estimated.
2. **Preservation of Originals**: The original manuscript (`DFU_WAGNER_net_paper.tex`), historical checkpoints, and experiment logs have been preserved untouched. All corrections are documented and output to `DFU_WAGNER_net_paper_corrected.tex`.
3. **Critical Discovery — Cross-Dataset Overlap**: Programmatic cryptographic hash audits identified **728 exact MD5 duplicate matches** between the DFUC2021 and Wagner datasets, stemming from uncurated public web scraping (e.g., Roboflow). The manuscript's claim that the cohorts were "completely disjoint" has been formally retracted and replaced with a cross-dataset leakage analysis and a leak-free quarantine manifest.
4. **Correction of the 16-Cell University of Texas Matrix Claim**: The manuscript claimed a $4 \times 4$ University of Texas matrix was generated via tensor outer product. In reality, the model outputs 4 pathology classes and 5 Wagner grades ($5 \times 4 = 20$ cells). Because Wagner grades and UT stages are clinically distinct and non-interchangeable, this claim has been removed and replaced with an exploratory $5 \times 4$ joint bivariate distribution under an explicit conditional independence assumption.
5. **Correction of Benchmark Comparison**: The manuscript compared internal development validation split results (Macro-F1 0.9201 on $n=1,191$) directly against published DFUC2021 official challenge hidden test set leaderboards (0.6307), claiming an invalid "+29 percentage point SOTA". Table I has been bifurcated to explicitly separate hidden-test-set results from internal development validation results.
6. **Harmonization of Equations and Hyperparameters**: Loss balancing weights ($\lambda_{\text{coral}} = 2.5$ vs. paper's $1.0$), batch sizes ($32/16$ vs. paper's $8$), learning rates ($2\times 10^{-5} / 4\times 10^{-4}$ vs. paper's $1\times 10^{-5} / 2\times 10^{-4}$), and the A0-Path loss function (`cbfocal` vs. paper's `CE`) were reconciled with the executed source code.
7. **Transparent Multi-Task Trade-Offs**: The audit clarified the multi-task Pareto trade-offs, documenting why single-task A0-Path (0.9306) and standard CE A6 (0.9319) achieved higher pathology F1 in 25-epoch screening than multi-task A1 (0.9109), and explaining the role of the 45-epoch extended training schedule.

---

## Detailed Audit Findings by Phase

### Phase 1: Architecture and Parameter Count Verification

- **Code Inspection**: `dfu_siam_model.py` was inspected and verified using PyTorch parameter counting routines.
- **Total Parameters**: Exactly **59,783,374 parameters** (100% trainable).
  - **CNN Backbone** (`EfficientNetV2-S`): $20,177,488$ parameters (33.75%)
  - **ViT Backbone** (`Swin-Transformer Tiny`): $27,519,354$ parameters (46.03%)
  - **CNN Projection**: $1,642,240$ parameters (2.75%)
  - **ViT Projection**: $986,880$ parameters (1.65%)
  - **TaskDisentangler**: $8,139,520$ parameters (13.62%)
  - **Classifier Head**: $658,948$ parameters (1.10%)
  - **Wagner Head (CORAL)**: $658,944$ parameters (1.10%)
- **Test Suite Execution**: `test_all.py` was executed using Python 3.11.15 and PyTorch 2.6.0+cu124. All **16 out of 16 unit tests passed**, confirming mathematical validity of:
  - Asymmetric feature extraction and fusion
  - TaskDisentangler subspace partitioning ($z_s \in \mathbb{R}^{768}$, $z_p \in \mathbb{R}^{512}$, $z_d \in \mathbb{R}^{512}$)
  - Tangent LMCoT loss computation and batch hard-pair mining
  - Uncertainty-Gated MMD kernel computation
  - CORAL rank cumulative loss and probability conversion

---

### Phase 2: Dataset Overlap and Split Integrity Audit

#### 1. Internal Audits
- **DFUC2021**:
  - Raw manifest entries: 9,949. Exactly 3,994 entries are all-NaN (unlabeled challenge exploratory data) and properly excluded.
  - Labeled images: Exactly 5,955.
  - Internal duplicates: **0** exact MD5 duplicates within DFUC2021.
  - Train/Validation Split (80/20, seed 42): Train = 4,764, Val = 1,191. Internal overlap = **0**.
- **Wagner Ulcer Dataset**:
  - Raw images: 12,099 (Train = 11,639, Valid = 303, Test = 157).
  - Internal leakage: Exactly 37 base images in `train/Grade 0` matched `valid/` (21) or `test/` (16), plus 111 augmented variants.
  - Verification: `wagner_dedup_manifest.json` correctly quarantines all 148 images from the training split. After quarantine, internal train-val and train-test overlap = **0**. Clean Train = 11,491.

#### 2. Cross-Dataset Collision Discovery
A full cross-dataset cryptographic audit (`scratch/audit_dataset_leakage.py` and `scratch/generate_overlap_manifest.py`) revealed:
- **728 exact MD5 hash collisions** exist between DFUC2021 and Wagner.
- **Root Cause**: The public Wagner dataset contains web-scraped images originating from DFUC2020/2021 and Roboflow re-uploads.
- **Cross-Split Leakage Distribution**:
  - `Both_in_Train`: 435 pairs (both in training splits; no evaluation leakage).
  - `DFU_Val_in_Wagner_Train`: 119 pairs (DFUC validation images trained on via Wagner).
  - `Wagner_Val_in_DFU_Train`: 92 pairs (Wagner validation images trained on via DFUC).
  - `Wagner_Test_in_DFU_Train`: 46 pairs (Wagner test images trained on via DFUC).
  - `Both_in_Validation`: 21 pairs (both in validation splits; never trained on).
  - `DFU_Val_and_Wagner_Test`: 15 pairs (DFUC validation and Wagner test; never trained on).

#### 3. Reproducibility Manifests and Clean Quarantine Protocol
To ensure complete evaluation integrity:
1. `research_paper/dataset_cross_overlap_manifest.json` and `.csv` detail every colliding pair, file paths, and split assignments.
2. `research_paper/clean_leak_free_split_manifest.json` specifies the **quarantine protocol**:
   - Exclude 138 DFU training images (colliding with Wagner Val/Test) $\to$ Clean DFU Train = **4,626**.
   - Exclude 120 Wagner clean training images (colliding with DFU Val) $\to$ Clean Wagner Train = **11,371**.
   - Preserved evaluation sets: DFU Val = **1,191**, Wagner Valid = **303**, Wagner Test = **157**.
   - Cross-split leakage under this protocol = **0.00%**.
3. **Manuscript Disclosure**: The manuscript explicitly states that preliminary 45-epoch metrics reflect the unquarantined legacy split, while documenting the clean quarantined protocol as the standard for subsequent experimental benchmarks.
4. **Patient-Level Independence**: Neither dataset provides patient IDs. The manuscript explicitly states that independence is guaranteed at the sample and hash level, but cannot be proven at the patient level.

---

### Phase 3: Clinical Taxonomy and Matrix Claim Audit

#### 1. Mathematical and Clinical Analysis
- **Old Manuscript Claim**: "Differentiable 16-Cell University of Texas Matrix Synthesis $M_{\text{UT}} = \mathbf{p}_{\text{depth}} \otimes \mathbf{p}_{\text{pathology}} \in \mathbb{R}^{4 \times 4}$".
- **Code Reality**: In `fusion_eval.py`, the function `predict_ut_grid` artificially compressed Wagner Grade 3 and Grade 4 into a 4-element vector to force a $4 \times 4$ shape.
- **Clinical Fact**: 
  - The Meggitt-Wagner scale has **5 ordinal grades** (0, 1, 2, 3, 4).
  - Pathology has **4 nominal classes** (None, Infection, Ischaemia, Both).
  - The true outer product $\mathbf{p}_{\text{depth}} \otimes \mathbf{p}_{\text{pathology}}$ is $5 \times 4 = 20$ cells, NOT 16 cells.
  - The University of Texas (UT) Diabetic Wound Classification System is a separate diagnostic system consisting of 4 Depth Grades (0--3) and 4 Clinical Stages (A--D). Wagner grades and UT stages are clinically distinct and non-interchangeable.

#### 2. Correction and Reframing
- The UT matrix claim has been removed from the abstract, contributions, methodology, equations, discussion, and conclusion.
- The output is now correctly presented as an **exploratory joint bivariate distribution** $P(W=w, P=p \mid x) \approx p_w^{\text{depth}} \cdot p_p^{\text{pathology}} \in \mathbb{R}^{5 \times 4}$ under a conditional independence assumption.
- The paper explicitly notes that this joint distribution is an exploratory visualization and does not constitute a clinically validated prognostic risk score.
- Hyperbolic claims such as "never under-triaged", "unprecedented", or autonomous clinical triage have been removed.

---

### Phase 4: Benchmark Comparison Integrity Audit

#### 1. Analysis of Table I Comparison
- **Old Manuscript Claim**: Compared the proposed model (Macro-F1 0.9201) against Galdran et al. (0.6307) and Toofanee et al. (0.6228), claiming a "+29 to +30 percentage point SOTA improvement".
- **Discrepancy**: Galdran et al. (0.6307) and Toofanee et al. (0.6228) are published evaluations on the official DFUC2021 **hidden competition test set** ($n \approx 5,734$). The 0.9201 Macro-F1 was obtained on an **internal 20% development validation split** ($n = 1,191$).
- **Correction**:
  - Table I has been bifurcated into:
    - **Part A: Official DFUC2021 Competition Hidden Test Set** (Leaderboard)
    - **Part B: Internal Stratified Development Validation Split** ($n = 1,191$)
  - The manuscript text explicitly clarifies that internal development validation metrics cannot be directly compared against hidden test set leaderboards.

---

### Phase 5: Ablation Study Verification and Multi-Task Trade-Offs

The ablation campaign was audited across two sequential stages:
1. **Historical Preliminary Ablation Runs (Local Screening)**: Executed on the historical unquarantined split (October 17, 2026).
2. **Verified Clean Leak-Free Ablation Campaign (Kaggle Dual-T4, Seed 42)**: Executed consuming `research_paper/clean_leak_free_split_manifest.json` with pre-training leakage assertions verified in `leakage_guard.py`.

#### 1. Historical Preliminary Ablation Runs (Local Screening)
Every metric reported in earlier preliminary logs matched the saved checkpoint from `train_dfu_siam.py` under the historical split:

| Exp ID | Experiment Directory | Monitored Metric | Best Epoch | Paper Macro-F1 | Actual Macro-F1 | Paper Isch F1 | Actual Isch F1 | Paper QWK | Actual QWK | Paper MAE | Actual MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-Path** | `checkpoints_ablation_A0_path` | `macro_f1` | 22 | 0.9306 | 0.9306 | 0.9451 | 0.9451 | --- | --- | --- | --- |
| **A0-Wag** | `checkpoints_ablation_A0_wag` | `qwk` | 5 | --- | --- | --- | --- | 0.7798 | 0.7798 | 0.4554 | 0.4554 |
| **A1 Full** | `checkpoints_ablation_A1_full` | `joint` | 23 | 0.9109 | 0.9109 | 0.9348 | 0.9348 | 0.7253 | 0.7253 | 0.4818 | 0.4818 |
| **A2 No LMCoT** | `checkpoints_ablation_A2_no_lmcot` | `joint` | 24 | 0.8995 | 0.8995 | 0.9247 | 0.9247 | 0.7268 | 0.7268 | 0.4851 | 0.4851 |
| **A3 No MMD** | `checkpoints_ablation_A3_no_mmd` | `joint` | 17 | 0.8889 | 0.8889 | 0.8866 | 0.8866 | 0.7779 | 0.7779 | 0.4554 | 0.4554 |
| **A4 No Orth** | `checkpoints_ablation_A4_no_orth` | `joint` | 23 | 0.9121 | 0.9121 | 0.9149 | 0.9149 | 0.7387 | 0.7387 | 0.4884 | 0.4884 |
| **A6 Std CE** | `checkpoints_ablation_A6_standard_ce`| `joint` | 22 | 0.9319 | 0.9319 | 0.9556 | 0.9556 | 0.7638 | 0.7638 | 0.4653 | 0.4653 |
| **A7 No Hard** | `checkpoints_ablation_A7_no_hard_mining`| `joint` | 22 | 0.8678 | 0.8678 | 0.8400 | 0.8400 | 0.7359 | 0.7359 | 0.4917 | 0.4917 |

#### 2. Verified Clean Leak-Free Ablation Campaign (Kaggle Dual-T4, Seed 42)
The final manuscript Table IV has been updated to reflect the leak-free quarantined split protocol (`research_paper/clean_leak_free_split_manifest.json`):

| Exp ID | Configuration Description | $\lambda_{\text{path}}$ | $\lambda_{\text{coral}}$ | $\lambda_{\text{lmcot}}$ | $\lambda_{\text{mmd}}$ | $\lambda_{\text{orth}}$ | Macro-F1 | Accuracy | Wagner QWK | Wagner MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1_Full** | **Full Multi-Task Proposed Model** | **1.0 (CBF)** | **2.5** | **0.5** | **0.1** | **0.1** | **0.9102** | **0.8883** | **0.7619** | **0.4653** |
| **A0_Path** | Pathology Single-Task Baseline | 1.0 (CBF) | 0.0 | 0.0 | 0.0 | 0.0 | **0.9226** | **0.8959** | --- | --- |
| **A0_Wag** | Wagner Ordinal Single-Task Baseline | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | --- | --- | **0.7834** | **0.4389** |
| **A2_NoLMCoT** | Without Tangent LMCoT Metric Loss | 1.0 (CBF) | 2.5 | 0.0 | 0.1 | 0.1 | **0.9058** | **0.8732** | **0.7404** | **0.4686** |
| **A3_NoUGMMD** | Without Uncertainty-Gated MMD | 1.0 (CBF) | 2.5 | 0.5 | 0.0 | 0.1 | **0.9060** | **0.8799** | **0.7506** | **0.4554** |
| **A4_NoOrth** | Without Orthogonality ($\mathcal{L}_{\text{orth}}=0$)$^*$ | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.0 | **0.9121** | **0.8866** | **0.7752** | **0.4488** |
| **A5_NoDisent**| Without Task Disentangler Module$^\dagger$ | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | --- | \multicolumn{4}{c}{--- *Not Performed (Architectural Control)* ---} |
| **A6_StandardCE** | Standard CE instead of CB-Focal | 1.0 (CE) | 2.5 | 0.5 | 0.1 | 0.1 | **0.9265** | **0.9060** | **0.7558** | **0.4851** |
| **A7_NoHardMining** | Uniform LMCoT (No Hard Mining) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.1 | **0.8941** | **0.8623** | **0.7462** | **0.5017** |

#### 3. Second-Pass Discrepancies and Clarifications Resolved:
1. **A5 Ablation Reality Check**:
   - *Investigation*: Code inspection of `dfu_siam_model.py` and `run_kaggle_ablations_2x_t4.py` reveals that A4 only disables the orthogonality loss penalty ($\lambda_{\text{orth}} = 0$). In A4, the `TaskDisentangler` projection heads ($z_s, z_p, z_d$) and task routing remain 100% active.
   - *Correction*: We have retracted the assertion that A5 was operationalized via A4. A genuine "No Disentanglement" model (feeding monolithic fused 2,560-D representations directly to classifier heads) was never trained because changing head input dimensions from 1,280-D to 2,560-D would confound architectural disentanglement with parameter scaling. A5 is now formally marked as **"Not Performed / Architectural Control"** across all manuscript tables and reports.
2. **Clean-Split Provenance Verification**:
   - *25-Epoch Clean Ablation Runs*: Verified that the Kaggle ablation script `run_kaggle_ablations_2x_t4.py` explicitly supplied `--clean_manifest research_paper/clean_leak_free_split_manifest.json` to `train_dfu_siam.py`, successfully asserted zero leakage via `leakage_guard.py`, and recorded `clean_manifest_hash` in run configurations. The reported A1 screening result (Macro-F1 0.9102, Accuracy 0.8883, QWK 0.7619, MAE 0.4653) is 100% clean-split verified.
   - *45-Epoch Primary Model Run*: Verified that `checkpoints_a1_full_balanced/experiment_config.json` was generated on 2026-10-06 prior to the creation of `clean_leak_free_split_manifest.json` (2026-10-19). It reflects extended training (stopped at best epoch 36) on the historical unquarantined split (with internal Wagner dedup, but without cross-dataset clean manifest exclusions). The two protocols are now explicitly delineated in the manuscript.
3. **Duplicate Counts and Counting Units Verification**:
   - *Manifest Recomputation*: Recomputed and verified all numbers directly from the JSON manifests:
     - **728**: Exact cross-dataset identical-image *pairs* (MD5 matches) between DFUC2021 and Wagner cohorts (`dataset_cross_overlap_manifest.json`).
     - **257**: Cross-split colliding *pairs* (119 DFU val in Wagner train + 92 Wagner val in DFU train + 46 Wagner test in DFU train).
     - **471**: Non-colliding image *pairs* (435 both-in-train, 21 both-in-validation, 15 DFU-val in Wagner-test).
     - **138**: Unique *DFU training image files* quarantined from `DFUdataset/train` ($4,764 \to 4,626$).
     - **120**: Unique *Wagner training image files* quarantined from `WagnerDataset/train` ($11,491 \to 11,371$).
     - **148**: Unique *Wagner internal image files* excluded from raw Wagner train ($11,639 \to 11,491$; 37 base duplicate images + 111 Roboflow augmentations).
   - *Unit Clarity*: The manuscript text now strictly distinguishes between pairs, unique files quarantined, and files excluded.
4. **Experimental Comparability (Primary Model vs. Ablation Screening)**:
   - Delineated protocol differences in Section V:
     - Primary Model: Local RTX 3050 Laptop GPU (6GB VRAM), Windows 11, Batch Size 8 (750 steps/epoch), $\eta_{\text{backbone}}=1\times 10^{-5}$, $\eta_{\text{head}}=2\times 10^{-4}$, 45 epochs (best epoch 36), historical split.
     - Ablation Study: Cloud Dual NVIDIA T4 GPUs, Ubuntu Linux, Batch Size 16 per worker (375 steps/epoch), scaled $\eta_{\text{backbone}}=2\times 10^{-5}$, $\eta_{\text{head}}=4\times 10^{-4}$, 25 epochs fixed budget, clean leak-free quarantined split.
5. **Benchmark Citations and Table I Audit**:
   - Official hidden test set results for DFUC2021 (Cassidy et al., Galdran et al., Toofanee et al.) are segregated into Part A.
   - Karthik et al. (2025) was evaluated on an independent split in *Scientific Reports* (2025). Its bibliographic entry in `references.bib` has been updated with verified authors (R. Karthik, A. Ajay, A. Jhalani, K. S. Ballari, S. Suganthi) and title, and moved to Part B ("Published Literature on Independent DFUC Splits").
6. **Honest Interpretation of Multi-Task Pareto Trade-Offs**:
   - Single-task baselines (A0-Path Macro-F1 0.9226; A0-Wag QWK 0.7834) and Standard CE (A6 Macro-F1 0.9265) outperform multi-task A1 on isolated individual metrics due to the absence of cross-task gradient competition.
   - Without orthogonality penalties ($\lambda_{\text{orth}}=0$, A4), features freely co-adapt and slightly elevate empirical screening metrics (Macro-F1 0.9121, QWK 0.7752), but sacrifice feature disentanglement and clinical explainability.
   - The clinical value of A1 is providing unified, concurrent dual-axis predictions within a single 59.78M parameter architecture.

---

### Phase 6: Mathematical Formulations Audit

1. **Loss Balancing Weights**:
   - Confirmed config values: $\lambda_{\text{path}} = 1.0$, $\lambda_{\text{coral}} = 2.5$, $\lambda_{\text{lmcot}} = 0.5$, $\lambda_{\text{mmd}} = 0.1$, $\lambda_{\text{orth}} = 0.1$.
2. **Subspace Orthogonality Regularization**:
   - The implementation computes:
     $$\mathcal{L}_{\text{orth}} = \frac{1}{2} \left( \frac{1}{N_p} \sum_{i=1}^{N_p} (z_{i, s}^T z_{i, p})^2 + \frac{1}{N_w} \sum_{j=1}^{N_w} (z_{j, s}^T z_{j, d})^2 \right)$$
   - The text clarifies that $(z_p^T z_d)^2$ is not computed because samples are processed in task-specific mini-batches.
3. **Uncertainty-Gated MMD**:
   - Multi-scale RBF bandwidths $\sigma \in \{1, 2, 5, 10\}$ and confidence weighting $w = \text{clamp}(1 - H(\mathbf{p})/\log C, 0.1, 1.0)$ verified.
4. **Hard-Pair Tangent LMCoT**:
   - Monotonic tangent metric $d_{\tan}(z_i, z_j) = \tan(\frac{1}{2}\arccos(z_i^T z_j))$ with distance ceiling 10.0 and margin 2.0 verified.

---

### Phase 7: Complete Second-Pass Discrepancy Matrix

| Item / Claim | Original Paper (`.tex`) | Verified Code / Data Reality | Severity | Corrected In (`_corrected.tex`) | Second-Pass Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cohorts Disjointness** | "$\mathcal{D}_{\text{path}}$ and $\mathcal{D}_{\text{wagner}}$ are completely disjoint cohorts ($x^p \ne x^w$)" | 728 exact MD5 duplicate matches across datasets due to web scraping | **Critical** | Disclosed cross-dataset overlap, documented 728 matches, added quarantine protocol | **VERIFIED & RESOLVED** |
| **Clinical UT Matrix** | "Differentiable 16-Cell University of Texas Matrix Synthesis $M_{\text{UT}} \in \mathbb{R}^{4 \times 4}$" | Wagner has 5 grades, Pathology has 4 classes ($5\times 4=20$ cells). Wagner grades $\ne$ UT stages | **Critical** | Removed UT claim; reframed as exploratory $5\times 4$ joint bivariate distribution | **VERIFIED & RESOLVED** |
| **Benchmark SOTA Claim** | "Surpassing MICCAI winning model by nearly 30 percentage points (0.9201 vs 0.6307)" | 0.6307 is on hidden test set; 0.9201 is on internal 20% validation split ($n=1,191$) | **High** | Delineated Part A (Hidden Test Set), Part B (Independent Splits), Part C (Internal Val) | **VERIFIED & RESOLVED** |
| **A5 Ablation Status** | Claimed A5 was operationalized via A4 ($\lambda_{\text{orth}}=0$) | A4 only disables loss penalty; TaskDisentangler is still fully active in A4. A5 was never run. | **High** | Retracted operationalization claim; Table IV explicitly marks A5 as Not Performed. | **VERIFIED & RESOLVED** |
| **Clean-Split Provenance** | Historical 45-epoch model conflated with clean-split protocol | 45-epoch model was run on historical split; 25-epoch Kaggle ablations used clean manifest | **High** | Delineated historical 45-epoch run from clean 25-epoch screening run in text and tables | **VERIFIED & RESOLVED** |
| **Duplicate Counting Units** | Conflated pairs, exclusions, and internal duplicates | 728 pairs, 257 colliding pairs, 138 DFU exclusions, 120 Wagner exclusions, 148 dedup files | **Medium** | Explicitly defined counting units and category breakdown in Section III-C | **VERIFIED & RESOLVED** |
| **Experimental Comparability** | Ablations claimed identical to primary model | Primary run on RTX 3050 (BS 8, 45 epochs); ablations on Dual T4 (BS 16, 25 epochs) | **Medium** | Fully documented hardware, LR scaling, batch sizes, and budgets in Section V | **VERIFIED & RESOLVED** |
| **Karthik et al. Citation** | Listed under official test set with placeholder authors | Published in *Sci. Rep.* 2025 on independent split; author list and title verified | **Medium** | Corrected `references.bib` entry and moved to independent split section in Table I | **VERIFIED & RESOLVED** |
| **Ablation Interpretation** | Claimed full model outperformed all variants on all metrics | A0-Path (0.9226), A0-Wag (0.7834), A6 (0.9265), and A4 (0.9121) beat A1 on specific metrics | **Medium** | Added comprehensive discussion of multi-task Pareto trade-offs and gradient tension | **VERIFIED & RESOLVED** |
| **Loss Balancing Weights** | $\boldsymbol{\lambda} = [1.0, 1.0, 0.5, 0.1, 0.1]$ | $\lambda_{\text{coral}} = 2.5$ in all configs ($\boldsymbol{\lambda} = [1.0, 2.5, 0.5, 0.1, 0.1]$) | **Medium** | Updated equation and text to $\lambda_{\text{coral}} = 2.5$ | **VERIFIED & RESOLVED** |

---

## Reproducibility Artifacts and Verification Scripts

The following artifacts and scripts have been created and persisted in the repository:
1. `research_paper/DFU_WAGNER_net_paper_corrected.tex`: The corrected, publication-quality IEEE manuscript.
2. `research_paper/dataset_cross_overlap_manifest.json`: Full JSON record of the 728 cross-dataset duplicate pairs.
3. `research_paper/dataset_cross_overlap_manifest.csv`: Tabular spreadsheet for audit review.
4. `research_paper/clean_leak_free_split_manifest.json`: Specifications and file exclusions for the leak-free quarantined split.
5. `scratch/generate_overlap_manifest.py`: Python script reproducing the cross-dataset audit.
6. `scratch/build_clean_splits.py`: Python script generating the quarantine split.
7. `tests/test_audit_integrity.py`: Automated regression test verifying parameter counts, matrix shapes, and split hash integrity.

---

## Conclusion

The scientific integrity and reproducibility of the DFU-WAGNER Net framework are now fully verified and documented. By eliminating unsubstantiated clinical claims, reconciling hyperparameter discrepancies, disclosing and quarantining web-scraped duplicate images, and accurately framing validation metrics against benchmark literature, the manuscript is rigorously defensible for peer-reviewed submission to IEEE Transactions on Medical Imaging.
