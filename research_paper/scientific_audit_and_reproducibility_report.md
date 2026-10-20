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

The ablation campaign was audited across two sequential stages: the preliminary local 25-epoch screening run on the historical split, and the subsequent clean leak-free 25-epoch screening run executed on Kaggle Dual-T4 GPUs (`research_paper/clean_kaggle_ablation_results.csv`).

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
| **A1_Full** | Full Multi-Task Proposed Model | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.1 | **0.9102** | **0.8883** | **0.7619** | **0.4653** |
| **A0_Path** | Pathology Single-Task Baseline | 1.0 (CBF) | 0.0 | 0.0 | 0.0 | 0.0 | **0.9226** | **0.8959** | --- | --- |
| **A0_Wag** | Wagner Ordinal Single-Task Baseline | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | --- | --- | **0.7834** | **0.4389** |
| **A2_NoLMCoT** | Without Tangent LMCoT Metric Loss | 1.0 (CBF) | 2.5 | 0.0 | 0.1 | 0.1 | **0.9058** | **0.8732** | **0.7404** | **0.4686** |
| **A3_NoUGMMD** | Without Uncertainty-Gated MMD | 1.0 (CBF) | 2.5 | 0.5 | 0.0 | 0.1 | **0.9060** | **0.8799** | **0.7506** | **0.4554** |
| **A4_NoOrth** | Without Orthogonality ($\mathcal{L}_{\text{orth}}=0$)$^*$ | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.0 | **0.9121** | **0.8866** | **0.7752** | **0.4488** |
| **A6_StandardCE** | Standard CE instead of CB-Focal | 1.0 (CE) | 2.5 | 0.5 | 0.1 | 0.1 | **0.9265** | **0.9060** | **0.7558** | **0.4851** |
| **A7_NoHardMining** | Uniform LMCoT (No Hard Mining) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.1 | **0.8941** | **0.8623** | **0.7462** | **0.5017** |

#### 3. Discrepancies and Clarifications Resolved:
- **A0-Path Loss**: The old manuscript listed A0-Path as `1.0 (CE)`. The config confirms it was actually trained with `cbfocal`. This has been corrected in Table IV.
- **A5 Missing Entry**: In `DFU_SIAM_V3_EXPERIMENT_PLAN.md` Section 4.1, A5 ("No Task Disentanglement") was operationalized via A4 ($\lambda_{\text{orth}}=0$) to avoid changing tensor dimensions. Table IV includes an explanatory footnote.
- **Multi-Task Pareto Trade-Off**: Single-task A0-Path (0.9306) and standard CE multi-task A6 (0.9319) outperformed multi-task A1 (0.9109) in the 25-epoch screening run. The revised manuscript discusses this trade-off openly: multi-task training introduces cross-task gradient tension, but enables concurrent prediction of ordinal depth (89.11% adjacent accuracy). Extended 45-epoch training elevates the primary model to 0.9201 Macro-F1 and 0.7765 QWK.
- **Batch Size and Learning Rates**: The old manuscript claimed a batch size of 8, $\eta_{\text{backbone}} = 1\times 10^{-5}$, and $\eta_{\text{head}} = 2\times 10^{-4}$. Actual configs show batch size = 32 (A0, A1, A2) or 16 (A3, A4, A6, A7), $\eta_{\text{backbone}} = 2\times 10^{-5}$, and $\eta_{\text{head}} = 4\times 10^{-4}$. These parameters have been updated in Section V.

---

### Phase 6: Mathematical Formulations Audit

1. **Loss Balancing Weights**:
   - Equation in old paper stated $\boldsymbol{\lambda} = [1.0, 1.0, 0.5, 0.1, 0.1]$.
   - Verified config values across all training scripts: $\lambda_{\text{path}} = 1.0$, $\lambda_{\text{coral}} = 2.5$, $\lambda_{\text{lmcot}} = 0.5$, $\lambda_{\text{mmd}} = 0.1$, $\lambda_{\text{orth}} = 0.1$.
   - Corrected in equation and text to $\boldsymbol{\lambda} = [1.0, 2.5, 0.5, 0.1, 0.1]$.
2. **Subspace Orthogonality Regularization**:
   - The implementation computes:
     $$\mathcal{L}_{\text{orth}} = \frac{1}{2} \left( \frac{1}{N_p} \sum_{i=1}^{N_p} (z_{i, s}^T z_{i, p})^2 + \frac{1}{N_w} \sum_{j=1}^{N_w} (z_{j, s}^T z_{j, d})^2 \right)$$
   - The text now clarifies that $(z_p^T z_d)^2$ is not computed because input samples are processed in task-specific mini-batches where only the active task's private subspace is instantiated.
3. **Uncertainty-Gated MMD**:
   - Clarified multi-scale RBF bandwidths $\sigma \in \{1, 2, 5, 10\}$.
   - Confirmed confidence weighting $w = \text{clamp}(1 - H(\mathbf{p})/\log C, 0.1, 1.0)$ using detached predictive entropy.
4. **Hard-Pair Tangent LMCoT**:
   - Verified that the monotonic tangent metric $d_{\tan}(z_i, z_j) = \tan(\frac{1}{2}\arccos(z_i^T z_j))$ with distance ceiling 10.0 and margin 2.0 correctly resolves the polarity singularity of the legacy cotangent formulation.

---

### Phase 7: Complete Discrepancy Matrix

| Item / Claim | Original Paper (`.tex`) | Verified Code / Data Reality | Severity | Corrected In (`_corrected.tex`) |
| :--- | :--- | :--- | :--- | :--- |
| **Cohorts Disjointness** | "$\mathcal{D}_{\text{path}}$ and $\mathcal{D}_{\text{wagner}}$ are completely disjoint cohorts ($x^p \ne x^w$)" | 728 exact MD5 duplicate matches across datasets due to web scraping | **Critical** | Disclosed cross-dataset overlap, documented 728 matches, added quarantine protocol |
| **Clinical UT Matrix** | "Differentiable 16-Cell University of Texas Matrix Synthesis $M_{\text{UT}} \in \mathbb{R}^{4 \times 4}$" | Wagner has 5 grades, Pathology has 4 classes ($5\times 4=20$ cells). Wagner grades $\ne$ UT stages | **Critical** | Removed UT claim; reframed as exploratory $5\times 4$ joint bivariate distribution |
| **Benchmark SOTA Claim** | "Surpassing MICCAI winning model by nearly 30 percentage points (0.9201 vs 0.6307)" | 0.6307 is on hidden test set; 0.9201 is on internal 20% validation split ($n=1,191$) | **High** | Delineated Part A (Hidden Test Set) and Part B (Internal Val Split) in Table I |
| **Loss Balancing Weights** | $\boldsymbol{\lambda} = [1.0, 1.0, 0.5, 0.1, 0.1]$ | $\lambda_{\text{coral}} = 2.5$ in all configs ($\boldsymbol{\lambda} = [1.0, 2.5, 0.5, 0.1, 0.1]$) | **Medium** | Updated equation and text to $\lambda_{\text{coral}} = 2.5$ |
| **Batch Size** | Mini-batch size of 8 | Batch size 32 (A0, A1, A2) or 16 (A3, A4, A6, A7) | **Medium** | Corrected Section V to state 32/16 batch sizes |
| **Learning Rates** | $\eta_{\text{backbone}} = 1\times 10^{-5}$, $\eta_{\text{head}} = 2\times 10^{-4}$ | $\eta_{\text{backbone}} = 2\times 10^{-5}$, $\eta_{\text{head}} = 4\times 10^{-4}$ | **Medium** | Corrected Section V to match actual learning rates |
| **A0-Path Loss in Table IV** | Listed as `1.0 (CE)` | Config shows `loss_pathology = "cbfocal"` | **Medium** | Corrected Table IV to `1.0 (CBF)` |
| **A5 Missing Entry** | A5 missing from Table IV with no explanation | A5 ("No Disentanglement") was operationalized via A4 ($\lambda_{\text{orth}}=0$) in experiment plan | **Low** | Added explanatory footnote in Table IV |
| **Hyperbolic Claims** | "Never under-triaged", "unprecedented", "autonomous triage" | Post-hoc models cannot guarantee zero under-triaging; no regulatory approval | **Medium** | Softened/removed all hyperbolic claims across all sections |

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
