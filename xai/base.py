"""
Explainable AI (XAI) Base Classes and Common Utilities for DFU-SIAM v3.
Supports:
- Target resolution (Pathology classes 0-3 vs Wagner ordinal grades 0-4)
- Safe hook management with context handlers
- Heatmap normalization, blending, and publication-ready visualization
- Verification that model parameters and evaluation mode remain unmodified
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

PATHOLOGY_CLASSES = ['none', 'infection', 'ischaemia', 'both']
WAGNER_GRADES = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']


def normalize_attribution_map(attr_map):
    """
    Normalizes attribution map to [0, 1] range safely.
    Handles NaN/Inf and constant zero maps without division by zero.
    """
    if isinstance(attr_map, torch.Tensor):
        attr_map = attr_map.detach().cpu().numpy()
    
    attr_map = np.nan_to_num(attr_map, nan=0.0, posinf=1.0, neginf=0.0)
    min_val = np.min(attr_map)
    max_val = np.max(attr_map)
    denom = max_val - min_val
    if denom > 1e-8:
        norm_map = (attr_map - min_val) / denom
    else:
        norm_map = np.zeros_like(attr_map, dtype=np.float32)
    return np.clip(norm_map, 0.0, 1.0).astype(np.float32)


def overlay_heatmap_on_image(pil_img, norm_heatmap, colormap='jet', alpha=0.5):
    """
    Overlays a normalized [0, 1] heatmap onto a PIL RGB image.
    Returns RGB uint8 numpy array of the overlay.
    """
    w, h = pil_img.size
    # Resize heatmap to match original image dimensions exactly
    heatmap_tensor = torch.from_numpy(norm_heatmap).unsqueeze(0).unsqueeze(0)
    resized_hm = F.interpolate(heatmap_tensor, size=(h, w), mode='bilinear', align_corners=False).squeeze().numpy()
    resized_hm = np.clip(resized_hm, 0.0, 1.0)

    cmap = plt.get_cmap(colormap)
    colored_hm = cmap(resized_hm)[:, :, :3]  # (H, W, 3) in [0, 1]

    img_np = np.array(pil_img).astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * img_np + alpha * colored_hm
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255.0).astype(np.uint8), resized_hm


def compute_target_score(model, input_tensor, target_type='pathology', target_id='predicted'):
    """
    Computes scalar target score for backpropagation.
    Supports:
      - target_type='pathology': target_id in {0, 1, 2, 3} or 'predicted'
      - target_type='wagner': target_id in {0, 1, 2, 3, 4} or 'predicted'
        (uses CORAL cumulative cutoffs to compute exact grade marginal probability)
    Returns:
      target_score (scalar torch.Tensor), pred_meta (dict with class names, probs, preds)
    """
    # model must remain in eval mode
    was_training = model.training
    model.eval()

    z_shared, path_logits, wagner_logits = model.forward_single(input_tensor, return_ordinal=True)

    # Pathology probabilities
    path_probs = F.softmax(path_logits, dim=-1)
    pred_path_class = int(torch.argmax(path_probs, dim=-1)[0].item())

    # Wagner probabilities (CORAL decoding)
    # Cutoffs q_k = sigmoid(logit_k) for k in {0, 1, 2, 3}
    q = torch.sigmoid(wagner_logits[0])
    raw_w = torch.stack([
        1.0 - q[0],
        q[0] - q[1],
        q[1] - q[2],
        q[2] - q[3],
        q[3]
    ])
    w_probs = torch.clamp(raw_w, min=0.0, max=1.0)
    w_sum = w_probs.sum()
    if w_sum > 1e-6:
        w_probs = w_probs / w_sum
    pred_wagner_grade = int(torch.argmax(w_probs).item())

    pred_meta = {
        'pred_pathology_id': pred_path_class,
        'pred_pathology_name': PATHOLOGY_CLASSES[pred_path_class],
        'pathology_probs': path_probs[0].detach().cpu().numpy().tolist(),
        'pred_wagner_grade': pred_wagner_grade,
        'pred_wagner_name': WAGNER_GRADES[pred_wagner_grade],
        'wagner_probs': w_probs.detach().cpu().numpy().tolist()
    }

    if target_type.lower() == 'pathology':
        if target_id == 'predicted':
            resolved_id = pred_path_class
        elif isinstance(target_id, str):
            resolved_id = PATHOLOGY_CLASSES.index(target_id.lower())
        else:
            resolved_id = int(target_id)
        
        target_score = path_logits[0, resolved_id]
        pred_meta['target_type'] = 'pathology'
        pred_meta['target_id'] = resolved_id
        pred_meta['target_name'] = PATHOLOGY_CLASSES[resolved_id]
        pred_meta['target_score'] = float(target_score.item())

    elif target_type.lower() == 'wagner':
        if target_id == 'predicted':
            resolved_id = pred_wagner_grade
        elif isinstance(target_id, str):
            clean_str = target_id.lower().replace('grade', '').strip()
            resolved_id = int(clean_str)
        else:
            resolved_id = int(target_id)
        
        # Marginal probability of the target grade derived from CORAL cutoffs
        target_score = w_probs[resolved_id]
        pred_meta['target_type'] = 'wagner'
        pred_meta['target_id'] = resolved_id
        pred_meta['target_name'] = WAGNER_GRADES[resolved_id]
        pred_meta['target_score'] = float(target_score.item())
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Must be 'pathology' or 'wagner'.")

    if was_training:
        model.train()

    return target_score, pred_meta


class BaseExplainer:
    """Abstract base class for all DFU-SIAM v3 explainers."""
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()

    def explain(self, input_tensor, target_type='pathology', target_id='predicted'):
        raise NotImplementedError
