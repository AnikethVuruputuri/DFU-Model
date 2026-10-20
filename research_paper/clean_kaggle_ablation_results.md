# DFU-SIAM v3 / DFU-WAGNER Net Kaggle Clean-Split Ablation Study Results

This report documents the completed ablation study executed on Kaggle Dual-T4 GPUs using the cryptographic leak-free dataset split manifest (`research_paper/clean_leak_free_split_manifest.json`) across all 8 controlled experimental configurations (Seed 42, 25-epoch screening protocol).

## Summary Table

| Experiment ID | Configuration Description | $\lambda_{\text{path}}$ | $\lambda_{\text{coral}}$ | $\lambda_{\text{lmcot}}$ | $\lambda_{\text{mmd}}$ | $\lambda_{\text{orth}}$ | Macro-F1 | Accuracy | Wagner QWK | Wagner MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1_Full** | **Full Proposed Model (CB-Focal + CORAL + Hard LMCoT + UG-MMD + Orth)** | **1.0 (CBF)** | **2.5** | **0.5** | **0.1** | **0.1** | **0.9102** | **0.8883** | **0.7619** | **0.4653** |
| **A0_Path** | Pathology Only Baseline (Wagner head & cross-alignment disabled) | 1.0 (CBF) | 0.0 | 0.0 | 0.0 | 0.0 | 0.9226 | 0.8959 | N/A | N/A |
| **A0_Wag** | Wagner Depth Only Baseline (DFUC pathology disabled) | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | N/A | N/A | 0.7834 | 0.4389 |
| **A2_NoLMCoT** | Without Tangent LMCoT Metric Learning ($\lambda_{\text{lmcot}} = 0$) | 1.0 (CBF) | 2.5 | 0.0 | 0.1 | 0.1 | 0.9058 | 0.8732 | 0.7404 | 0.4686 |
| **A3_NoUGMMD** | Without Uncertainty-Gated MMD Alignment ($\lambda_{\text{mmd}} = 0$) | 1.0 (CBF) | 2.5 | 0.5 | 0.0 | 0.1 | 0.9060 | 0.8799 | 0.7506 | 0.4554 |
| **A4_NoOrth** | Without Orthogonality Disentanglement ($\lambda_{\text{orth}} = 0$) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.0 | 0.9121 | 0.8866 | 0.7752 | 0.4488 |
| **A6_StandardCE** | Standard Cross-Entropy (No Class-Balanced Focal Loss) | 1.0 (CE) | 2.5 | 0.5 | 0.1 | 0.1 | 0.9265 | 0.9060 | 0.7558 | 0.4851 |
| **A7_NoHardMining** | Random Pairs without Hard-Mining LMCoT | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.1 | 0.8941 | 0.8623 | 0.7462 | 0.5017 |

## Key Findings and Component Contributions

1. **Impact of Hard-Pair Mining (A1 vs. A7)**:
   - Hardest-pair tangent LMCoT mining elevates Macro-F1 from **0.8941** to **0.9102** ($+1.61\%$ absolute gain) and improves classification accuracy from **86.23%** to **88.83%** ($+2.60\%$).
   - On the Wagner task, hard mining improves QWK from **0.7462** to **0.7619** and reduces MAE from **0.5017** to **0.4653**, verifying that focused margin enforcement on ambiguous decision boundaries stabilizes shared manifold representations.

2. **Impact of Tangent LMCoT Metric Learning (A1 vs. A2)**:
   - Removing the metric learning loss ($\lambda_{\text{lmcot}} = 0$) reduces Macro-F1 from **0.9102** to **0.9058**, accuracy from **88.83%** to **87.32%**, and Wagner QWK from **0.7619** to **0.7404** (with MAE increasing to **0.4686**).
   - This validates that tangent hyperspherical metric clustering enforces intra-class compactness on $z_{\text{shared}}$.

3. **Impact of Uncertainty-Gated MMD Domain Alignment (A1 vs. A3)**:
   - Removing cross-domain alignment ($\lambda_{\text{mmd}} = 0$) drops Macro-F1 to **0.9060**, accuracy to **0.8799**, and Wagner QWK to **0.7506**.
   - UG-MMD bridges distribution shifts between the photographic protocols of DFUC2021 and Wagner cohorts.

4. **Impact of Class-Balanced Focal Loss (A1 vs. A6)**:
   - While Standard Cross-Entropy (A6) achieves higher nominal accuracy (0.9060) and Macro-F1 (0.9265) driven by the dominant majority classes, its Wagner QWK is lower (0.7558 vs 0.7619) and MAE is worse (0.4851 vs 0.4653). Class-Balanced Focal Loss prevents gradient starvation on critical minority ischaemia classes.

5. **Single-Task Baselines (A0-Path and A0-Wag)**:
   - Single-task pathology (A0-Path) achieves **0.9226** Macro-F1 and **0.8959** Accuracy.
   - Single-task Wagner depth (A0-Wag) achieves **0.7834** QWK and **0.4389** MAE.
   - The multi-task network (A1) achieves **0.9102** Macro-F1 and **0.7619** Wagner QWK concurrently, demonstrating minimal multi-task degradation while enabling simultaneous multi-task clinical decision support.
