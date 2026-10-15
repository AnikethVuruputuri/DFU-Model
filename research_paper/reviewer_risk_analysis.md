# Peer Reviewer Risk Analysis: Top 15 Rejection Vulnerabilities

**Role:** Senior Peer Reviewer for IEEE T-PAMI / IEEE T-MI / Medical Image Analysis / MICCAI.  
**Review Mindset:** Rigorous, unsparing, evidence-driven peer critique.

---

## The Top 15 Rejection Risks

### 1. Incompatible Validation vs. Benchmark Test Split Comparison
* **Reviewer Concern:** The authors report an impressive Macro-F1 of 0.9201 on DFUC2021 and claim superiority over Galdran et al. (0.6307) and Toofanee et al. (0.6228). However, the literature scores were evaluated on the **official hidden challenge test set ($n=5,734$)**, whereas the authors' score is measured on a **local 20% validation split ($n=1,191$)** carved out of the training set. Comparing validation metrics against hidden competition test benchmarks is scientifically invalid.
* **Severity:** 🔴 **CRITICAL**
* **Evidence:** `checkpoints_a1_full_balanced/temp_val_split.csv` has 1,191 rows. `output/dfu_siam_predictions.csv` contains test set predictions, but ground truth test labels are withheld by challenge organizers.
* **Remediation Experiment:** Explicitly label local results as *Local Validation Split (n=1,191)*, benchmark all reproduced baselines on the **identical local split**, and submit model predictions to the official DFUC benchmark server if open.

---

### 2. Lack of Patient-Level Disjoint Splitting (Potential Patient Data Leakage)
* **Reviewer Concern:** Neither the DFUC2021 dataset nor the public Roboflow Wagner dataset provides patient identifiers. Consequently, multiple photographs taken from the same patient across consecutive hospital visits may exist in both the training and validation partitions, causing significant data leakage and inflated validation performance.
* **Severity:** 🔴 **CRITICAL**
* **Evidence:** `dataset_audit.py` confirmed 0 duplicate filenames, but 52 near-duplicate perceptual hash clusters in DFUC2021, and 731 exact image matches between DFUC2021 and the Wagner dataset. Patient metadata is completely absent in public CSVs.
* **Remediation Experiment:** Explicitly state in the *Limitations* section that patient IDs are absent from public challenge releases. Perform cross-split perceptual hash clustering to enforce strict patient-proxy cluster separation, re-evaluating the model on cluster-disjoint splits.

---

### 3. Missing Component Ablation Suite (Unverified Claims of Multi-Task Synergy)
* **Reviewer Concern:** The paper proposes five interrelated components: Hard-Pair LMCoT ($\mathcal{L}_{\text{lmcot}}$), CORAL ordinal loss ($\mathcal{L}_{\text{coral}}$), Uncertainty-Gated MMD ($\mathcal{L}_{\text{mmd}}$), Orthogonality regularization ($\mathcal{L}_{\text{orth}}$), and Class-Balanced Focal loss ($\mathcal{L}_{\text{path}}$). Yet, the paper provides zero completed ablation runs ($A0, A2, A3, A4, A6, A7$). Without empirical ablations, the claim that each component contributes positively is entirely unsupported.
* **Severity:** 🔴 **CRITICAL**
* **Evidence:** `ablation_comparison.csv` indicates that $A0\_Path$, $A0\_Wag$, $A2\_NoLMCoT$, $A3\_NoUGMMD$, $A4\_NoOrth$, $A6\_StandardCE$, and $A7\_NoHardMining$ are **NOT RUN**.
* **Remediation Experiment:** Execute the full 7-experiment ablation suite specified in `run_experiments.py` for at least 25–30 epochs to empirically populate $\Delta\text{Macro-F1}$ and $\Delta\text{QWK}$.

---

### 4. Single Random Seed Execution (Absence of Statistical Robustness)
* **Reviewer Concern:** All reported primary results stem from a single training run with `seed=42`. Deep learning on long-tailed medical imaging datasets exhibits non-trivial stochastic variance across random seed initializations. Without multi-seed error bounds, the findings lack scientific reliability.
* **Severity:** 🟠 **HIGH**
* **Evidence:** `experiment_config.json` confirms only `seed: 42` was executed. No seed 123 or seed 2026 checkpoints exist on disk.
* **Remediation Experiment:** Train the primary configuration across seeds 42, 123, and 2026 using `run_experiments.py` and report mean $\pm$ standard deviation for all primary metrics.

---

### 5. Absence of Statistical Significance Testing
* **Reviewer Concern:** The manuscript lacks hypothesis testing (McNemar's test, paired permutation tests, Wilcoxon signed-rank tests, or DeLong tests for AUROC). The authors cannot prove whether performance differences over baselines are statistically significant or merely random fluctuations.
* **Severity:** 🟠 **HIGH**
* **Evidence:** No statistical significance tests exist in the codebase; no confidence intervals or p-values were established for comparative models.
* **Remediation Experiment:** Perform bootstrap resampling ($B=1,000$) on the validation predictions to compute 95% confidence intervals for Macro-F1 and QWK; execute McNemar's test against baseline predictions.

---

### 6. Small and Imbalanced Wagner Validation Set ($n=303$)
* **Reviewer Concern:** The Wagner validation split contains only 303 images, with severe class skew (only 21 images for Grade 0 and 40 for Grade 3). In Grade 3, the model achieves a precision of only 0.3542 and an F1 of 0.3864. Drawing strong clinical conclusions from 40 test samples is clinically precarious.
* **Severity:** 🟠 **HIGH**
* **Evidence:** Confusion matrix indicates that out of 40 true Grade 3 ulcers, only 17 were correctly classified (17 mispredicted as Grade 2, 5 as Grade 4, 1 as Grade 1).
* **Remediation Experiment:** Report 5-fold stratified cross-validation on the combined Wagner dataset (11,491 images) to ensure every fold evaluates over 2,000+ test samples.

---

### 7. Unverified Clinical Validity of the 16-Cell University of Texas Matrix
* **Reviewer Concern:** The authors synthesize a $4 \times 4$ University of Texas (UT) wound staging matrix by taking the outer product of independent pathology probabilities and Wagner depth cutoffs. However, the model was trained on disjoint cohorts and never validated on a true ground-truth UT-annotated clinical cohort. Calling this a validated clinical decision aid is an overreach.
* **Severity:** 🟠 **HIGH**
* **Evidence:** Neither DFUC2021 nor the Wagner dataset contains concurrent multi-axial UT labels. The zero-shot UT matrix is purely mathematical.
* **Remediation Experiment:** Tone down clinical claims; explicitly frame the 16-cell matrix as a *computational exploratory proof-of-concept*, rather than a clinically validated diagnostic tool.

---

### 8. Moderate Ordinal Error Spread on Wagner Grading ($\text{Accuracy} = 68.65\%$)
* **Reviewer Concern:** While the Quadratic Weighted Kappa is 0.7765 and adjacent-grade accuracy is 91.42%, the exact nominal grade accuracy is only 68.65%, and severe errors ($\ge 2$ grades) occur in 8.58% of cases. In a surgical context, misclassifying Grade 1 as Grade 3 or Grade 4 could lead to unnecessary amputation workups.
* **Severity:** 🟡 **MEDIUM**
* **Evidence:** Off-diagonal errors $\ge 2$ grades total 26 samples ($8.58\%$ of validation set).
* **Remediation Experiment:** Incorporate cost-sensitive ordinal penalties or asymmetric loss weights that heavily penalize depth underestimation.

---

### 9. Potential Overfitting to Training Artifacts in Public Web Datasets
* **Reviewer Concern:** The Wagner dataset originates from Roboflow web scraping and clinical photography with variable backgrounds (bedsheets, rulers, hospital floors). The model could be picking up background shortcuts rather than true tissue excavation.
* **Severity:** 🟡 **MEDIUM**
* **Evidence:** XAI Grad-CAM and SHAP figures demonstrate that in misclassified cases (Figure B), background skin texture and non-ulcer boundaries receive non-zero attribution.
* **Remediation Experiment:** Evaluate attribution localization against bounding boxes or ulcer masks; run background masking stress tests.

---

### 10. Lack of Comparison Against Recent SOTA Vision Foundation Models
* **Reviewer Concern:** The study benchmarks against 2021–2023 literature, but omits modern biomedical foundation models such as BiomedCLIP, Med-SAM, or fine-tuned DINOv2.
* **Severity:** 🟡 **MEDIUM**
* **Evidence:** All comparisons are against ResNeXt50, standard DeiT, and original DFU-SIAM.
* **Remediation Experiment:** Add an experimental baseline evaluating zero-shot / linear-probe performance of a modern vision foundation model (e.g., BiomedCLIP).

---

### 11. Unreproduced Literature Baselines on Identical Splits
* **Reviewer Concern:** Baselines from Toofanee et al., Galdran et al., and Cassidy et al. are cited directly from their papers rather than retrained on the exact same local split used for DFU-SIAM v3.
* **Severity:** 🟡 **MEDIUM**
* **Evidence:** `baseline_comparison.csv` indicates literature baselines are copied from published challenge tables.
* **Remediation Experiment:** Train standalone standard ResNet50 and Swin-Tiny classifiers on `temp_train_split.csv` and report their performance on `temp_val_split.csv`.

---

### 12. Training Paused Prior to Full Convergence (Epoch 36 / 45)
* **Reviewer Concern:** The balanced model was scheduled for 45 epochs but stopped at Epoch 36. Reviewers may question whether the model reached optimal convergence or was prematurely interrupted.
* **Severity:** 🟢 **LOW**
* **Evidence:** `best_checkpoint.pth` corresponds to Epoch 36; `learning_curves.json` logs 9 resumed epochs.
* **Remediation Experiment:** Demonstrate via validation learning curves that the joint validation metric plateaued between epochs 30–36, confirming natural convergence.

---

### 13. Computational Footprint Exceeds Edge Deployment Targets (59.8M Parameters)
* **Reviewer Concern:** At 59.8M parameters and 228 MB, the full model cannot easily be deployed on mobile clinical edge devices (e.g., bedside point-of-care tablets).
* **Severity:** 🟢 **LOW**
* **Evidence:** `benchmark_lightweight.py` confirms that while the 59.8M model takes 532 ms/step, the 35.2M Swin-Tiny model and 12.5M MobileViT model achieve up to 161 images/sec.
* **Remediation Experiment:** Present the 35.2M Swin-Tiny variant in the main paper as the primary edge-deployable alternative.

---

### 14. Lack of Prospective Clinical Reader Study
* **Reviewer Concern:** The paper claims diagnostic utility for podiatrists and wound care nurses, but includes no clinical reader study comparing human clinician accuracy with vs. without AI assistance.
* **Severity:** 🟢 **LOW**
* **Evidence:** No human reader study data is collected.
* **Remediation Experiment:** Clarify that this manuscript focuses on *computational representation learning and algorithmic multi-task alignment*, explicitly recommending prospective reader studies for future clinical trials.

---

### 15. Ambiguity Regarding XAI Heatmaps as Diagnostic Proof
* **Reviewer Concern:** Authors often over-interpret Grad-CAM heatmaps, claiming the network "understands pathology." Reviewers reject papers that equate saliency maps with clinical causal reasoning.
* **Severity:** 🟢 **LOW**
* **Evidence:** `docs/XAI.md` and `xai/unified.py` explicitly state that saliency maps are gradient attributions, not segmentation boundaries or proofs of medical reasoning.
* **Remediation Experiment:** Maintain strict adherence to the cautionary clinical disclosures already established in `docs/XAI.md`.
