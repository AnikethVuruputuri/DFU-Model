"""
Explainable AI (XAI) Module for DFU-SIAM v3.
Provides model-agnostic and architecture-specific explainability tools:
- Grad-CAM (CNN, ViT, Fused)
- Transformer-specific attribution (Swin-Tiny)
- Integrated Gradients (Path attribution)
- SHAP (Partition/Superpixel Shapley values)
- Unified XAIExplainer interface
- Faithfulness evaluation (AOPC Deletion and Insertion tests)
"""

from xai.base import (
    BaseExplainer,
    PATHOLOGY_CLASSES,
    WAGNER_GRADES,
    overlay_heatmap_on_image,
    normalize_attribution_map,
)
from xai.gradcam import GradCAMExplainer
from xai.transformer import TransformerExplainer
from xai.integrated_gradients import IntegratedGradientsExplainer
from xai.shap_explainer import SHAPExplainer
from xai.evaluation import run_deletion_test, run_insertion_test
from xai.unified import XAIExplainer

__all__ = [
    "BaseExplainer",
    "PATHOLOGY_CLASSES",
    "WAGNER_GRADES",
    "overlay_heatmap_on_image",
    "normalize_attribution_map",
    "GradCAMExplainer",
    "TransformerExplainer",
    "IntegratedGradientsExplainer",
    "SHAPExplainer",
    "run_deletion_test",
    "run_insertion_test",
    "XAIExplainer",
]
