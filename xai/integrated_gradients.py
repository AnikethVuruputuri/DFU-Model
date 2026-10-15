"""
Integrated Gradients Explainer for DFU-SIAM v3.
Computes path-integrated gradients from a configurable baseline (black, blurred, mean).
Path:
  IG_i(x) = (x_i - x'_i) * integral_{alpha=0}^1 (dF(x' + alpha*(x - x')) / dx_i) d_alpha
Supports both Pathology logits and Wagner ordinal probability targets.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np

from xai.base import BaseExplainer, compute_target_score, normalize_attribution_map


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Path-integrated gradients explainer with configurable baselines.
    """
    def __init__(self, model, steps=25, baseline_type='black', device=None):
        super().__init__(model, device)
        self.steps = steps
        self.baseline_type = baseline_type

    def _create_baseline(self, input_tensor):
        """Creates baseline image based on baseline_type."""
        if self.baseline_type == 'black':
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == 'white':
            return torch.ones_like(input_tensor)
        elif self.baseline_type == 'blur':
            # 11x11 average blur
            kernel = torch.ones(1, 1, 11, 11, device=self.device) / 121.0
            blurred = []
            for c in range(input_tensor.shape[1]):
                ch = input_tensor[:, c:c+1]
                bl = F.conv2d(ch, kernel, padding=5)
                blurred.append(bl)
            return torch.cat(blurred, dim=1)
        else: # 'mean'
            mean_val = input_tensor.mean(dim=(2, 3), keepdim=True)
            return mean_val.expand_as(input_tensor)

    def explain(self, input_tensor, target_type='pathology', target_id='predicted'):
        start_t = time.time()
        input_tensor = input_tensor.to(self.device)
        self.model.eval()

        baseline = self._create_baseline(input_tensor)

        # 1. Determine target metadata on original image first
        with torch.no_grad():
            _, pred_meta = compute_target_score(
                self.model, input_tensor, target_type=target_type, target_id=target_id
            )
        resolved_type = pred_meta['target_type']
        resolved_id = pred_meta['target_id']

        # 2. Generate alphas along straight line path
        alphas = torch.linspace(0.0, 1.0, self.steps, device=self.device)
        grad_accum = torch.zeros_like(input_tensor)

        # Process in batches to avoid GPU VRAM exhaustion on laptop GPUs
        chunk_size = 5
        for i in range(0, self.steps, chunk_size):
            chunk_alphas = alphas[i:i+chunk_size]
            interp_list = [baseline + a * (input_tensor - baseline) for a in chunk_alphas]
            batch_interp = torch.cat(interp_list, dim=0).requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            _, p_logits, w_logits = self.model.forward_single(batch_interp, return_ordinal=True)

            if resolved_type == 'pathology':
                score = p_logits[:, resolved_id].sum()
            else: # Wagner
                q = torch.sigmoid(w_logits)
                w_probs = torch.stack([
                    1.0 - q[:, 0],
                    q[:, 0] - q[:, 1],
                    q[:, 1] - q[:, 2],
                    q[:, 2] - q[:, 3],
                    q[:, 3]
                ], dim=1)
                score = w_probs[:, resolved_id].sum()

            score.backward()
            grad_accum += batch_interp.grad.sum(dim=0, keepdim=True)

        avg_grads = grad_accum / float(self.steps)
        delta = input_tensor - baseline
        ig_attributions = delta * avg_grads  # (1, 3, H, W)

        # Aggregate across color channels
        ig_spatial = ig_attributions.abs().sum(dim=1).squeeze(0)  # (H, W)
        ig_heatmap = normalize_attribution_map(ig_spatial)

        runtime = time.time() - start_t
        pred_meta['runtime_sec'] = float(runtime)
        pred_meta['xai_method'] = 'integrated_gradients'
        pred_meta['ig_steps'] = self.steps
        pred_meta['baseline_type'] = self.baseline_type

        return {
            'ig_heatmap': ig_heatmap,
            'meta': pred_meta
        }
