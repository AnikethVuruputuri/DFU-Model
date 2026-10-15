# DFU-WAGNER net Research Paper Package: Overview & Status

**Project:** DFU-WAGNER net (Diabetic Foot Ulcer – Shared Intelligence and Alignment Model, Version 3)  
**Target Publication Venues:** IEEE Transactions on Medical Imaging (TMI) / IEEE Journal of Biomedical and Health Informatics (JBHI) / IEEE ISBI / MICCAI  
**Working Directory:** `research_paper/`  
**Current Date:** October 2026  
**Status:** Manuscript Structure Complete up to Final Empirical Results Insertion

---

## 1. Executive Summary

This directory contains the complete, publication-quality research paper draft and supporting scholarly documentation for **DFU-WAGNER net**. In strict adherence to scientific integrity:
- **No experimental results have been invented, estimated, or fabricated.**
- Architectural details and parameter counts reflect the **exact source code implementation** in the repository (specifically the primary 59,783,374-parameter candidate model).
- Clear, standardized placeholders (`[FINAL TEST RESULTS PENDING]`, `[CONFUSION MATRIX AND PER-CLASS METRICS PENDING]`, etc.) are established across the manuscript, tables, and figures.
- When the ongoing training and evaluation runs complete, empirical metrics can be directly slotted into the designated sections.

---

## 2. Inventory of Prepared Files

| File Name | Functional Description | Status |
| :--- | :--- | :---: |
| [`DFU_SIAM_v3_paper.tex`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/DFU_SIAM_v3_paper.tex) | **Primary LaTeX Manuscript Source Code** (IEEEtran format). Camera-ready source code complete with mathematical equations, 8 booktabs tables, figure specifications, and structured results placeholders. Ready for Overleaf / ArXiv. | **Complete (Placeholders Marked)** |
| [`main.tex`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/main.tex) | **Overleaf / Compiler Entrypoint**. Standard root LaTeX file linking `DFU_SIAM_v3_paper.tex`. | **Complete** |
| [`references.bib`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/references.bib) | **BibTeX Database**. 35 fully verified academic citations with official DOIs and complete metadata. | **Complete & Verified** |
| [`DFU_SIAM_v3_paper_draft.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/DFU_SIAM_v3_paper_draft.md) | **Markdown Reference Draft** (IEEE formatting mirror). Parallel markdown version for instant local reading. | **Complete (Placeholders Marked)** |
| [`literature_review.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/literature_review.md) | **Comprehensive Literature Survey and Prior Art Analysis**. In-depth review of Topics A–H, granular analysis of original DFU-SIAM (Toofanee et al., *IEEE Access*, 2023), 22-paper Literature Review Table, verified Research Gaps, and exact Implemented Contributions. | **Complete & Verified** |
| [`references.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/references.md) | **Academic Bibliography (IEEE Style)**. 35 verified scholarly references complete with authors, article titles, journals/conferences, volumes, pages, years, and official DOIs/identifiers. | **Complete & Verified** |
| [`figures_plan.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/figures_plan.md) | **Publication Figures Specifications (Figures 1–7)**. Complete architectural ASCII flowcharts, tensor dimensions, Grad-CAM layer targets, loss interaction graphs, and figure placeholders. | **Complete & Specified** |
| [`tables_plan.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/tables_plan.md) | **Publication Tables Framework (Tables I–VIII)**. Formats, verified dataset and architectural counts, and empirical evaluation layouts. | **Complete & Formatted** |
| [`experiments_pending.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/experiments_pending.md) | **Scientific Execution Tracker & CLI Commands**. Checklist of pending evaluations (A0 baselines, A1 candidate, A2–A7 ablations, 3-seed reproducibility), commands, and slotting protocol. | **Complete & Actionable** |
| [`README.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/README.md) | This document: project status, verified metrics vs. placeholders, and step-by-step completion guide. | **Active** |

---

## 3. Verified Numbers vs. Pending Empirical Results

### 3.1 Verified Values (Confirmed from Source Code & Local Audits)
* **Model Parameter Count (Primary 60M Candidate)**:
  - Total Parameters: **59,783,374** (100% trainable).
  - CNN Backbone (`tf_efficientnetv2_s.in21k_ft_in1k`): **20,177,488** (33.75%).
  - ViT Backbone (`swin_tiny_patch4_window7_224`): **27,519,354** (46.03%).
  - Symmetrical Projection Heads ($1,280 \to 1,280$ and $768 \to 1,280$): **2,629,120** (4.40%).
  - `TaskDisentangler` (3 parallel projection heads): **8,139,520** (13.62%).
  - Task Heads (Pathology 4-class + Wagner CORAL 4-cutoff): **1,317,892** (2.20%).
  - Memory Footprint: FP32 Weights: **228.06 MB** | FP16 Weights: **114.03 MB**.
  - Peak Training VRAM: **$\approx 4,310\text{ MB}$** under AMP FP16 at batch size 8.
* **DFUC2021 Pathology Dataset Statistics**:
  - Total rows in `train.csv`: **9,949**.
  - Unannotated rows filtered: **3,994** (40.14%).
  - Clean labeled training cohort: **5,955**.
  - Class distribution: `None`: 2,552 (42.85%), `Infection`: 2,555 (42.91%), `Ischaemia`: 227 (3.81%), `Both`: 621 (10.43%).
  - Stratified 80/20 split: Train: **4,764** images | Val: **1,191** images.
  - Majority-to-minority imbalance ratio: **11.26 : 1.00**.
* **Wagner Ulcer Dataset Statistics (Post-Deduplication)**:
  - Raw images on disk: **12,099**.
  - Excluded cross-split duplicate base images: **37**.
  - Excluded cross-split duplicate augmented images: **111**.
  - Clean training cohort: **11,491** images.
  - Preserved validation cohort: **303** images.
  - Preserved test cohort: **157** images.
  - Cross-split duplicate hashes remaining: **0** (100% hash independence).

### 3.2 Sections Containing Empirical Placeholders
The following sections in `DFU_SIAM_v3_paper_draft.md` contain placeholders to be populated upon experimental completion:
1. **Abstract**: `[FINAL TEST RESULTS TO BE INSERTED AFTER EXPERIMENTS]`
2. **Section 6.1 (Pathology Results)**: `[FINAL TEST RESULTS PENDING]`
3. **Section 6.2 (Per-Class Performance)**: `[CONFUSION MATRIX AND PER-CLASS METRICS PENDING]`
4. **Section 6.3 (Wagner Depth Results)**: `[FINAL WAGNER TEST RESULTS PENDING]`
5. **Section 6.4 (Ablation Results)**: `[ABLATION RESULTS PENDING]`
6. **Section 7 (Discussion)**: `[DISCUSSION FRAMEWORK: Comprehensive comparative discussion...]`
7. **Section 11 (Conclusion)**: `[FINAL CONCLUSION PARAGRAPH REGARDING EMPIRICAL VALIDATION METRICS...]`
8. **Tables Plan**: Tables IV, V, VI, and VII.
9. **Figures Plan**: Figures 4, 5, 6, and 7.

---

## 4. Step-by-Step Guide for Inserting Final Results

When the model training and evaluation script finishes:
1. Run standalone evaluation on the best checkpoint:
   ```bash
   python evaluate_checkpoint.py \
       --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
       --arch full \
       --data_dir DFUdataset \
       --wagner_dir WagnerDataset \
       --batch_size 16 \
       --output_dir output_eval_final
   ```
2. Open `output_eval_final/evaluation_report.json`:
   - Extract `macro_f1`, `accuracy`, `precision`, `recall`, and per-class F1 for `ischaemia`.
   - Extract Wagner `qwk`, `mae`, `adjacent_acc`, and `spearman_rho`.
3. Open [`DFU_SIAM_v3_paper_draft.md`](file:///c:/Users/srian/Desktop/WagnerDS/research_paper/DFU_SIAM_v3_paper_draft.md):
   - Replace `[FINAL TEST RESULTS TO BE INSERTED AFTER EXPERIMENTS]` in Abstract with the summary numbers.
   - Replace Table IV placeholders with the DFUC2021 metrics.
   - Replace Table VI placeholders with the 4-class per-class scores.
   - Replace Table VII placeholders with the Wagner ordinal metrics.
   - Populate Table V with ablation metrics once ablation runs conclude.
4. Render the confusion matrices into Figures 5 and 6 using `evaluate_checkpoint.py` outputs.
5. Generate multi-branch Grad-CAM figures using `dfu_siam_model.py::get_cam_target_layers()` for Figure 7.
