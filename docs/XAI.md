# Explainable AI (XAI) Module for DFU-SIAM v3

Comprehensive documentation for the Explainable AI (XAI) inference suite integrated into **DFU-SIAM v3** (Diabetic Foot Ulcer Siamese Cross-Taxonomy Disentangled Network).

---

## 1. System Overview & Core Objectives

The DFU-SIAM v3 XAI module provides **post-hoc, gradient-based, and perturbation-based model attribution** across both branches of the hybrid architecture:
- **CNN Branch**: EfficientNetV2-S (`conv_head`, spatial resolution $7 \times 7 \times 1280$)
- **Vision Transformer Branch**: Swin-Tiny (`swin_tiny_patch4_window7_224`, final stage window attention LayerNorm, token resolution $7 \times 7 \times 768$)
- **Dual-Branch Fusion**: Symmetrical combination ($0.5 \cdot \text{CNN} + 0.5 \cdot \text{ViT}$)
- **Disentangled Heads**: Multi-class Pathology (`None`, `Infection`, `Ischaemia`, `Both`) and Wagner Ordinal Depth Grading (Grades 0–4 via CORAL cumulative logits).

### Architecture Invariance Guarantee
> [!IMPORTANT]
> The XAI module operates **strictly on already-trained checkpoints during inference**. It introduces **zero modifications** to the model architecture:
> - No layers added, removed, or replaced.
> - Feature projections, disentangler matrices, and head weights remain **byte-for-byte unmodified**.
> - PyTorch forward and backward hooks are registered dynamically and **strictly cleaned up** upon completion.
> - Evaluation mode (`model.eval()`) and gradient isolation (`torch.no_grad()` where appropriate) are preserved.

---

## 2. Implemented XAI Methods

| Method | Target Scope | Mechanism | Spatial Granularity |
| :--- | :--- | :--- | :--- |
| **CNN Grad-CAM** | CNN Branch | Gradient-weighted activation map of `conv_head` | $7 \times 7 \to 224 \times 224$ (Bilinear) |
| **Transformer Attribution** | ViT Branch | Window-attention token gradient-energy map from Swin Stage 4 | $7 \times 7 \to 224 \times 224$ (Bilinear) |
| **Dual-Branch Fused** | Combined | Normalized linear combination of CNN & Transformer maps | $224 \times 224$ |
| **Integrated Gradients** | Full Network | Straight-line path Riemann integral $\int_0^1 \frac{\partial F(x' + \alpha(x-x'))}{\partial x} d\alpha$ | Pixel-level ($224 \times 224 \times 3$) |
| **Superpixel KernelSHAP** | Full Network | Partition-based Shapley value estimation using $14 \times 14$ grid masking | Region-level ($196$ superpixels) |

---

## 3. Mathematical Formulation of Targets

### A. Pathology Multi-Class Logits
For target pathology class $c \in \{0, 1, 2, 3\}$ (`none`, `infection`, `ischaemia`, `both`):
$$S_{\text{pathology}}(x) = z_c(x)$$
Where $z_c(x)$ is the raw unnormalized logit corresponding to class $c$.

### B. Wagner Ordinal Grades via CORAL Cutoffs
DFU-SIAM v3 uses CORAL (Consistent Rank Logits) to model ordinal ulcer depth across 5 grades ($0, 1, 2, 3, 4$) using 4 cumulative binary thresholds:
$$q_k(x) = \sigma(z_k^{\text{wagner}}(x)) = P(y > k), \quad k \in \{0, 1, 2, 3\}$$
The discrete grade probabilities are derived as:
$$P(y = 0) = 1 - q_0$$
$$P(y = k) = \max(0, q_{k-1} - q_k), \quad k \in \{1, 2, 3\}$$
$$P(y = 4) = q_3$$
When explaining a Wagner grade target $k$, the scalar score for backpropagation is set directly to the derived marginal probability $P(y = k)$. Gradients flow directly through the cumulative sigmoid cutoffs into both `z_shared` and `z_depth_private`.

---

## 4. CLI Tools & Usage

### 4.1 Single-Image Explanation (`explain_prediction.py`)

Generate full multi-method explanations for a single image:
```bash
python explain_prediction.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --image DFUdataset/DFUC2021_train/images/305971.jpg \
    --target pathology \
    --target_id predicted \
    --xai_method all \
    --output_dir outputs/xai/
```

Explain specific Wagner depth grade:
```bash
python explain_prediction.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --image WagnerDataset/valid/Grade\ 3/image_01.jpg \
    --target wagner \
    --target_id 3 \
    --xai_method gradcam
```

Compute quantitative faithfulness metrics (AOPC Deletion and Insertion AUC):
```bash
python explain_prediction.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --image DFUdataset/DFUC2021_train/images/303344.jpg \
    --xai_method gradcam \
    --run_faithfulness
```

### 4.2 Batch Cohort Evaluation (`explain_dataset.py`)

Run batch explanations across validation samples with failure analysis filtering:
```bash
# Analyze only misclassified samples to diagnose model errors
python explain_dataset.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --num_samples 20 \
    --filter incorrect_only \
    --xai_method gradcam \
    --output_dir outputs/xai_dataset_errors

# Cohort-wide faithfulness evaluation across correct predictions
python explain_dataset.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --num_samples 15 \
    --filter correct_only \
    --xai_method gradcam \
    --run_faithfulness \
    --output_dir outputs/xai_dataset_cohort
```

### 4.3 Publication Figure Generator (`generate_xai_figures.py`)

Generate publication-ready 300 DPI figures:
```bash
python generate_xai_figures.py \
    --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth \
    --output_dir outputs/xai_figures
```
Produces:
- `figure_a_correct_prediction.png`: Multi-method true-positive showcase.
- `figure_b_misclassification_analysis.png`: Diagnostic failure mode comparison (target: predicted vs true).
- `figure_c_minority_ischaemia.png`: Feature localization for scarce Ischaemia cases.
- `figure_d_wagner_depth_grading.png`: Attribution across ordinal wound depth severity.
- `figure_e_method_comparison.png`: Cross-method resolution and branch complementarity matrix.

---

## 5. Output Artifacts & JSON Specification

For each analyzed image, the system generates:
```text
outputs/xai/<image_name>/
    ├── original.png                     # Original input RGB image
    ├── gradcam_cnn.png                  # CNN (EfficientNetV2) overlay
    ├── gradcam_cnn_raw.png              # CNN raw colormapped heatmap
    ├── gradcam_vit.png                  # ViT overlay
    ├── gradcam_fused.png                # Dual-branch fused overlay
    ├── transformer_attribution.png      # Swin token attribution overlay
    ├── transformer_raw.png              # Swin raw heatmap
    ├── integrated_gradients.png         # Integrated Gradients overlay
    ├── integrated_gradients_raw.png     # Integrated Gradients raw map
    ├── shap.png                         # KernelSHAP superpixel overlay
    ├── shap_raw.png                     # KernelSHAP raw map
    ├── combined_summary.png             # Publication-ready side-by-side figure
    └── explanation.json                 # Comprehensive metadata and evaluation scores
```

### Explanation JSON Schema
```json
{
    "pred_pathology_id": 1,
    "pred_pathology_name": "infection",
    "pathology_probs": [0.03, 0.91, 0.04, 0.02],
    "pred_wagner_grade": 2,
    "pred_wagner_name": "Grade 2",
    "wagner_probs": [0.02, 0.12, 0.74, 0.09, 0.03],
    "target_type": "pathology",
    "target_id": 1,
    "target_name": "infection",
    "target_score": 3.842,
    "runtime_sec": 0.42,
    "xai_method": "all",
    "model_version": "v3",
    "model_architecture": "full",
    "layers_used": {
        "cnn": "model.backbone.cnn.conv_head",
        "vit": "model.backbone.vit.layers[-1].blocks[-1].norm1",
        "fused": "dual-branch symmetrical fusion (0.5 * CNN + 0.5 * ViT)",
        "transformer": "model.backbone.vit.layers[-1].blocks[-1].norm1",
        "integrated_gradients": "input_pixel_space",
        "shap": "superpixel_grid_14x14"
    },
    "xai_configuration": {
        "ig_steps": 25,
        "ig_baseline": "black",
        "shap_grid_size": 14,
        "shap_n_evals": 100
    },
    "faithfulness_evaluation": {
        "aopc_deletion_score": 0.3841,
        "insertion_auc": 0.6912
    }
}
```

---

## 6. Quantitative Faithfulness Metrics

### AOPC Deletion Test
Measures the decline in target logit/probability as the highest-attributed pixels are systematically masked by a local Gaussian blur:
$$\text{AOPC} = \frac{1}{K} \sum_{k=1}^K \left[ S(x) - S(x_{\setminus k}) \right]$$
A higher AOPC indicates higher fidelity: masking the attributed regions immediately destroys model confidence.

### Insertion Test
Starts from a blurred baseline and progressively introduces the highest-attributed pixels back into the image. Computes the Area Under the Curve (AUC): higher AUC signifies rapid recovery of classification confidence using minimal visual evidence.

---

## 7. Clinical & Scientific Caveats

> [!WARNING]
> 1. **Model Attribution $\neq$ Anatomical Segmentation**:
>    XAI heatmaps represent regions that mathematically influenced the artificial neural network's decision; they do **not** represent verified clinical wound boundaries, histological margins, or segmentation masks.
> 2. **Contextual Attention**:
>    The transformer branch often attends to peripheral foot anatomy (e.g., surrounding hallux or calcaneal skin) to establish spatial context or detect erythema, which is clinically valid context but outside the ulcer crater.
> 3. **Non-Diagnostic**:
>    These visualizations are research tools designed to audit model trustworthiness, identify shortcut learning, and assist debugging. They are **not approved medical diagnostic devices**.
