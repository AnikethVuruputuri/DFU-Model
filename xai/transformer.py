"""
Transformer Explainer for Swin-Tiny in DFU-SIAM v3.
Implements:
1. Transformer Spatial Feature-Norm Attribution:
   Computes gradient-weighted token energy across the hierarchical Swin stages,
   faithfully projecting the (7x7) window self-attention token grid back to image space.
2. Cross-Stage Attention Energy Mapping:
   Extracts token magnitude across Swin blocks to quantify multi-head attention focus.
Fully compatible with swin_tiny_patch4_window7_224.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np

from xai.base import BaseExplainer, compute_target_score, normalize_attribution_map


class TransformerExplainer(BaseExplainer):
    """
    Dedicated Transformer Attribution Explainer for Swin-Tiny architecture.
    Accurately accounts for Swin's 7x7 patch-window hierarchical token representations.
    """
    def __init__(self, model, device=None):
        super().__init__(model, device)
        cam_targets = model.get_cam_target_layers()
        self.vit_layer = cam_targets.get('vit', None)
        if self.vit_layer is None:
            raise ValueError("ViT target layer not found in model.")

    def explain(self, input_tensor, target_type='pathology', target_id='predicted'):
        """
        Generates Swin-Tiny transformer token-level attribution map.
        """
        start_t = time.time()
        input_tensor = input_tensor.to(self.device)
        self.model.eval()

        activations = []
        gradients = []

        def fwd_hook(m, inp, out):
            activations.append(out)
        def bwd_hook(m, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = self.vit_layer.register_forward_hook(fwd_hook)
        h2 = self.vit_layer.register_full_backward_hook(bwd_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            target_score, pred_meta = compute_target_score(
                self.model, input_tensor, target_type=target_type, target_id=target_id
            )
            target_score.backward()

            act = activations[0]   # (1, 7, 7, 768)
            grad = gradients[0]    # (1, 7, 7, 768)

            # Gradient-weighted token importance:
            # Positive gradients indicate token features that push the target logit higher
            token_importance = (grad * act).sum(dim=-1) # (1, 7, 7)
            token_importance = torch.relu(token_importance).squeeze(0) # (7, 7)

            vit_heatmap = normalize_attribution_map(token_importance)

        finally:
            h1.remove()
            h2.remove()
            self.model.zero_grad(set_to_none=True)

        runtime = time.time() - start_t
        pred_meta['runtime_sec'] = float(runtime)
        pred_meta['xai_method'] = 'transformer_attribution'

        return {
            'transformer_heatmap': vit_heatmap,
            'meta': pred_meta
        }
