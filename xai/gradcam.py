"""
Grad-CAM Explainer for CNN and Transformer Branches in DFU-SIAM v3.
Supports:
1. CNN Branch (EfficientNetV2 final conv_head / blocks)
2. Transformer Branch (Swin-Tiny final stage LayerNorm reshaped to 7x7 spatial tokens)
3. Fused Explanation (Normalized linear combination weighted by branch projections)
All hooks are strictly unregistered in finally blocks to eliminate memory leaks.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np

from xai.base import BaseExplainer, compute_target_score, normalize_attribution_map


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM Explainer for DFU-SIAM v3 hybrid CNN-Transformer architecture.
    """
    def __init__(self, model, device=None):
        super().__init__(model, device)
        cam_targets = model.get_cam_target_layers()
        self.cnn_target_layer = cam_targets.get('cnn', None)
        self.vit_target_layer = cam_targets.get('vit', None)

        if self.cnn_target_layer is None:
            raise ValueError("Could not dynamically locate CNN target layer in model.backbone.cnn")
        if self.vit_target_layer is None:
            raise ValueError("Could not dynamically locate ViT target layer in model.backbone.vit")

    def explain(self, input_tensor, target_type='pathology', target_id='predicted'):
        """
        Generates branch-level Grad-CAM heatmaps.
        Returns:
            dict containing:
              - 'cnn_heatmap': (H, W) normalized map
              - 'vit_heatmap': (H, W) normalized map
              - 'fused_heatmap': (H, W) normalized combined map
              - 'meta': runtime, prediction metadata, target information
        """
        start_t = time.time()
        input_tensor = input_tensor.to(self.device)
        self.model.eval()

        activations = {}
        gradients = {}

        def get_hook(key):
            def fwd_hook(module, inp, out):
                activations[key] = out
            def bwd_hook(module, grad_in, grad_out):
                gradients[key] = grad_out[0]
            return fwd_hook, bwd_hook

        h_cnn_fwd, h_cnn_bwd = get_hook('cnn')
        h_vit_fwd, h_vit_bwd = get_hook('vit')

        hook_handles = [
            self.cnn_target_layer.register_forward_hook(h_cnn_fwd),
            self.cnn_target_layer.register_full_backward_hook(h_cnn_bwd),
            self.vit_target_layer.register_forward_hook(h_vit_fwd),
            self.vit_target_layer.register_full_backward_hook(h_vit_bwd),
        ]

        try:
            self.model.zero_grad(set_to_none=True)
            target_score, pred_meta = compute_target_score(
                self.model, input_tensor, target_type=target_type, target_id=target_id
            )
            target_score.backward()

            # 1. Compute CNN Grad-CAM
            # CNN act: (1, C, H_c, W_c), grad: (1, C, H_c, W_c)
            cnn_act = activations['cnn']
            cnn_grad = gradients['cnn']
            cnn_weights = cnn_grad.mean(dim=(2, 3), keepdim=True)
            cnn_cam = torch.relu((cnn_weights * cnn_act).sum(dim=1)).squeeze(0)
            cnn_heatmap = normalize_attribution_map(cnn_cam)

            # 2. Compute ViT (Swin-Tiny) Grad-CAM
            # Swin-Tiny layer norm act: (1, 7, 7, C), grad: (1, 7, 7, C)
            vit_act = activations['vit']
            vit_grad = gradients['vit']
            if vit_act.dim() == 4:  # (B, H, W, C)
                vit_weights = vit_grad.mean(dim=(1, 2), keepdim=True)
                vit_cam = torch.relu((vit_weights * vit_act).sum(dim=-1)).squeeze(0)
            elif vit_act.dim() == 3:  # (B, N, C) for standard ViT
                vit_weights = vit_grad.mean(dim=1, keepdim=True)
                vit_cam = torch.relu((vit_weights * vit_act).sum(dim=-1)).squeeze(0)
                # Reshape square tokens if applicable
                side = int(np.sqrt(vit_cam.numel()))
                vit_cam = vit_cam.view(side, side)
            else:
                vit_cam = vit_act.mean(dim=-1).squeeze(0)

            vit_heatmap = normalize_attribution_map(vit_cam)

            # 3. Fused Dual-Branch Explanation
            # Upsample both to input spatial grid (224x224) and combine with equal branch weights
            # Both representations contribute symmetrically to self.fused_dim = self.proj_dim * 2
            t_cnn = torch.from_numpy(cnn_heatmap).unsqueeze(0).unsqueeze(0)
            t_vit = torch.from_numpy(vit_heatmap).unsqueeze(0).unsqueeze(0)
            target_size = (input_tensor.shape[2], input_tensor.shape[3])

            up_cnn = F.interpolate(t_cnn, size=target_size, mode='bilinear', align_corners=False).squeeze()
            up_vit = F.interpolate(t_vit, size=target_size, mode='bilinear', align_corners=False).squeeze()
            fused_cam = 0.5 * up_cnn + 0.5 * up_vit
            fused_heatmap = normalize_attribution_map(fused_cam)

        finally:
            # Strictly remove all hooks to prevent memory leaks and state pollution
            for h in hook_handles:
                h.remove()
            self.model.zero_grad(set_to_none=True)

        runtime = time.time() - start_t
        pred_meta['runtime_sec'] = float(runtime)
        pred_meta['xai_method'] = 'gradcam'

        return {
            'cnn_heatmap': cnn_heatmap,
            'vit_heatmap': vit_heatmap,
            'fused_heatmap': fused_heatmap,
            'meta': pred_meta
        }
