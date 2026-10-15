# Publication Figures Plan: DFU-WAGNER net

This document specifies the complete visual plan for Figures 1 through 7 for the IEEE-style manuscript draft. Figures requiring final empirical data contain structured placeholders to be populated upon experimental completion.

---

## Figure 1: Overall DFU-WAGNER net End-to-End Multi-Task Architecture

### Description
High-resolution architectural diagram depicting the dual-backbone input processing, symmetrical projection heads, fused 2,560-D representation, 3-way `TaskDisentangler`, selective task branching, and multi-task loss formulation.

### Tensor Flow and Modular Layout
```
Input Image x ∈ ℝ^(B × 3 × 224 × 224)
   │
   ├── Branch 1: EfficientNetV2-S (Conv Backbone) ──> Output: ℝ^(B × 1280)
   │        └── CNN Proj Head: Dropout(0.3) -> Linear(1280, 1280) -> BN -> GELU ──> ℝ^(B × 1280)
   │
   └── Branch 2: Swin-Transformer Tiny (ViT) ────────> Output: ℝ^(B × 768)
            └── ViT Proj Head: Dropout(0.3) -> Linear(768, 1280)  -> BN -> GELU ──> ℝ^(B × 1280)
                                                                                       │
   ┌───────────────────────────────────────────────────────────────────────────────────┘
   ▼
[Concatenation Layer] ──> Fused Feature Representation ℝ^(B × 2560)
   │
   ▼
[TaskDisentangler Module]
   ├── Shared Projector:  Linear(2560->1024) -> BN -> GELU -> Dropout -> Linear(1024->768) -> BN -> L2Norm ──> z_shared ∈ ℝ^(B × 768)
   ├── Path Projector:    Linear(2560->768)  -> BN -> GELU -> Dropout -> Linear(768->512)  -> BN -> L2Norm ──> z_path_private ∈ ℝ^(B × 512)
   └── Depth Projector:   Linear(2560->768)  -> BN -> GELU -> Dropout -> Linear(768->512)  -> BN -> L2Norm ──> z_depth_private ∈ ℝ^(B × 512)
   │
   ├──────────────────────────────────────────┬──────────────────────────────────────────┐
   ▼                                          ▼                                          ▼
[Selective Pathology Pathway]              [Selective Wagner Pathway]                 [Cross-Cohort Alignment]
Concat [z_shared; z_path_private]          Concat [z_shared; z_depth_private]         Shared Subspace z_shared ONLY
       ℝ^(B × 1280)                               ℝ^(B × 1280)                                    │
           │                                          │                                           ▼
[Pathology Classifier Head]                [Wagner CORAL Ordinal Head]                [Uncertainty-Gated MMD]
Linear(1280->512) -> BN -> ReLU            Linear(1280->512) -> BN -> ReLU            Multi-scale Gaussian RBF RKHS
-> Dropout(0.3) -> Linear(512->4)          -> Dropout(0.3) -> Linear(512->4)          Weighted by Detached Entropy
           │                                          │                               Confidence: w = 1 - H/log(C)
           ▼                                          ▼                                           │
4 Logits (None, Inf, Isch, Both)           4 Cumulative Rank Cutoffs                               ▼
           │                                          │                                       ℒ_UG-MMD
           ▼                                          ▼
      ℒ_CB-Focal                                   ℒ_CORAL
           │                                          │
           └──────────────────┬───────────────────────┘
                              ▼
                     [Orthogonality Loss]
           ℒ_orth = ‖z_shared · z_path_private‖² + ‖z_shared · z_depth_private‖²
                              │
                              ▼
                     [Metric Learning Loss]
           ℒ_Hard-LMCoT on z_shared (Hardest-positive / hardest-negative mining)
```

---

## Figure 2: CNN-Transformer Feature Fusion and Subspace Disentanglement

### Description
Detailed schematic highlighting the mathematical disentanglement of the fused 2,560-D vector into three orthogonal unit hyperspheres ($\mathbb{S}^{767}$ for $z_{\text{shared}}$, and $\mathbb{S}^{511}$ for $z_{\text{path\_private}}$ and $z_{\text{depth\_private}}$).

### Key Components Illustrated
1. **Symmetrical Projection Symmetry**: How EfficientNetV2-S ($1,280 \to 1,280$) and Swin-Tiny ($768 \to 1,280$) are normalized before concatenation.
2. **Subspace Separation**: Visual diagram showing how the 3 projectors enforce $z_{\text{shared}} \perp z_{\text{path\_private}}$ and $z_{\text{shared}} \perp z_{\text{depth\_private}}$.
3. **Information Barrier**: Explicit visual barrier proving $z_{\text{depth\_private}}$ has zero computational path to the pathology classifier, and $z_{\text{path\_private}}$ has zero path to the Wagner head.

---

## Figure 3: Multi-Task Loss Interaction and Metric Learning Framework

### Description
Mathematical diagram illustrating the five cooperating loss objectives:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{path}} \mathcal{L}_{\text{CB-Focal}} + \lambda_{\text{coral}} \mathcal{L}_{\text{CORAL}} + \lambda_{\text{lmcot}} \mathcal{L}_{\text{Hard-LMCoT}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{UG-MMD}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$
with default weights $\lambda = [1.0, 1.0, 0.5, 0.1, 0.1]$.

### Visual Elements
1. **Class-Balanced Focal Loss**: Curve showing effective sample weight $E_c = (1 - \beta^{n_c})/(1 - \beta)$ scaling up the gradient for the 3.8% ischaemia class while downweighting saturated majority classes.
2. **CORAL Binary Thresholding**: Visualization of the $K-1=4$ parallel hyperplanes on the 1D ordinal depth line.
3. **Tangent Metric Learning**: Angular distance plot demonstrating monotonic growth $\tan(\theta/2)$ without singularity, contrasting with the inverted cotangent bug.
4. **Uncertainty Gating**: Diagram showing how high-entropy (uncertain) predictions receive low MMD alignment weight $w \to 0.1$, while confident samples drive alignment $w \to 1.0$.

---

## Figure 4: Training Dynamics and Validation Learning Curves

### Description
Multi-panel training progression plots showing convergence across epochs.

### Panels
- **Panel 4A**: Total training loss vs. epoch.
- **Panel 4B**: Decomposition of loss components ($\mathcal{L}_{\text{path}}$, $\mathcal{L}_{\text{coral}}$, $\mathcal{L}_{\text{lmcot}}$, $\mathcal{L}_{\text{mmd}}$, $\mathcal{L}_{\text{orth}}$) across epochs.
- **Panel 4C**: DFUC2021 validation Macro-F1 progression comparing baseline vs. full proposed model.
- **Panel 4D**: Wagner validation Quadratic Weighted Kappa (QWK) and Mean Absolute Error (MAE) progression.

### Data Status
`[PLACEHOLDER: Curves to be plotted directly from output/learning_curves.json upon final multi-epoch completion]`

---

## Figure 5: Multi-Class Confusion Matrix for DFUC2021 Pathology Classification

### Description
A $4 \times 4$ normalized and raw count confusion matrix evaluated on the held-out validation set ($n = 1,191$).

### Class Layout
- Rows (True Labels): `None` ($n=510$), `Infection` ($n=511$), `Ischaemia` ($n=46$), `Both` ($n=124$).
- Columns (Predicted Labels): `None`, `Infection`, `Ischaemia`, `Both`.

### Target Visual Indicators
- Diagonal dominance showing true positive rates.
- Specific off-diagonal inspection of `Ischaemia` vs. `None` to confirm suppression of false negatives.
- Inspection of `Infection` vs. `Both` boundary.

### Data Status
`[PLACEHOLDER: Confusion matrix to be rendered from evaluate_checkpoint.py execution on best_checkpoint.pth]`

---

## Figure 6: Ordinal Confusion Matrix and Error Distribution for Wagner Grading

### Description
A $5 \times 5$ ordinal confusion matrix evaluated on the preserved held-out Wagner validation cohort ($n = 303$) and test cohort ($n = 157$).

### Class Layout
- Grades 0, 1, 2, 3, 4 along true and predicted axes.

### Visual Elements
- Quadratic Weighted Kappa color intensity overlay highlighting penalization of off-diagonal deviations.
- Adjacent accuracy envelope ($|\hat{y} - y| \le 1$) marked with dotted white borders, illustrating that clinical misclassifications are confined to adjacent grades.

### Data Status
`[PLACEHOLDER: Ordinal matrix to be rendered from evaluate_checkpoint.py execution on best_checkpoint.pth]`

---

## Figure 7: Model Attribution and Explainability via Multi-Branch Grad-CAM

### Description
Visual attribution heatmaps generated for representative clinical cases across all 4 pathology classes and 5 Wagner grades.

### Architectural Attribution Targets
- **CNN Branch**: EfficientNetV2-S `conv_head` / `blocks[-1]` (capturing localized edge textures and erythema margins).
- **Transformer Branch**: Swin-Transformer Tiny `layers[-1].blocks[-1].norm1` (capturing global contextual foot morphology).
- **Fused Overlay**: Combined attribution map overlaying the original RGB photograph.

### Clinical Disclosure Note
Caption will explicitly state: *"Attribution heatmaps represent post-hoc gradient saliency maps and do NOT constitute wound segmentation masks, histological boundaries, or causal clinical proofs."*

### Image Selection Specification
- Sample 1: Superficial clean ulcer (`None`, Grade 1).
- Sample 2: Cellulitis with erythema margin (`Infection`, Grade 2).
- Sample 3: Pale necrotic ischaemic wound (`Ischaemia`, Grade 2).
- Sample 4: Advanced gangrenous neuro-ischaemic ulcer (`Both`, Grade 4).

### Data Status
`[PLACEHOLDER: Visual heatmaps to be rendered using get_cam_target_layers() from dfu_siam_model.py upon final checkpoint selection]`
