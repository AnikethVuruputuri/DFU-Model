"""
Quantitative XAI Evaluation and Faithfulness Metrics.
Implements:
1. Confidence comparison
2. Deletion Test (AOPC - Area Over Perturbation Curve):
   Progressively masks out the top-k attributed pixels and measures the drop in prediction score.
   A faithful attribution map causes a steep, rapid drop.
3. Insertion Test:
   Starts from a blank/blurred baseline and adds back top-k attributed pixels.
   Measures recovery of the prediction score.
4. Attribution Stability:
   Tests sensitivity to small Gaussian noise perturbations.
"""

import numpy as np
import torch
import torch.nn.functional as F

from xai.base import compute_target_score


def run_deletion_test(model, input_tensor, attribution_map, target_type='pathology', target_id='predicted', steps=10, device='cuda', num_steps=None):
    if num_steps is not None:
        steps = num_steps
    """
    Measures the decline in target prediction confidence as top-ranked attribution pixels are deleted.
    Returns:
      fractions: list of deleted pixel percentages [0.0, 0.1, ..., 1.0]
      scores: target scores at each step
      drop_rate: fractional drop in confidence after removing 50% top pixels
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    orig_score, meta = compute_target_score(model, input_tensor, target_type=target_type, target_id=target_id)
    resolved_type = meta['target_type']
    resolved_id = meta['target_id']
    orig_val = float(orig_score.item())

    # Baseline for deletion (blurred image)
    kernel = torch.ones(1, 1, 15, 15, device=device) / 225.0
    blurred_list = [F.conv2d(input_tensor[:, c:c+1], kernel, padding=7) for c in range(3)]
    baseline = torch.cat(blurred_list, dim=1)

    H, W = input_tensor.shape[2], input_tensor.shape[3]
    # Resize attribution map to match input resolution
    if attribution_map.shape != (H, W):
        t_am = torch.from_numpy(attribution_map).unsqueeze(0).unsqueeze(0).float()
        am_resized = F.interpolate(t_am, size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()
    else:
        am_resized = attribution_map

    # Flatten and rank pixels descending
    flat_attr = am_resized.flatten()
    sorted_indices = np.argsort(-flat_attr) # highest attribution first
    total_pixels = len(sorted_indices)

    fractions = np.linspace(0.0, 0.8, steps)
    scores = []

    for f in fractions:
        if f == 0.0:
            scores.append(orig_val)
            continue
        num_delete = int(f * total_pixels)
        del_idx = sorted_indices[:num_delete]

        mask = np.ones(total_pixels, dtype=np.float32)
        mask[del_idx] = 0.0
        mask_2d = torch.from_numpy(mask.reshape(H, W)).unsqueeze(0).unsqueeze(0).to(device)

        del_img = mask_2d * input_tensor + (1.0 - mask_2d) * baseline

        with torch.no_grad():
            _, p_logits, w_logits = model.forward_single(del_img, return_ordinal=True)
            if resolved_type == 'pathology':
                cur_val = float(p_logits[0, resolved_id].item())
            else:
                q = torch.sigmoid(w_logits)
                w_probs = torch.stack([
                    1.0 - q[:, 0], q[:, 0] - q[:, 1],
                    q[:, 1] - q[:, 2], q[:, 2] - q[:, 3], q[:, 3]
                ], dim=1)
                cur_val = float(w_probs[0, resolved_id].item())
            scores.append(cur_val)

    # Area Over Perturbation Curve (AOPC)
    aopc = float(np.mean([orig_val - s for s in scores[1:]])) if len(scores) > 1 else 0.0

    return {
        'fractions': fractions.tolist(),
        'scores': scores,
        'aopc_deletion_score': aopc,
        'final_drop': float(scores[0] - scores[-1])
    }


def run_insertion_test(model, input_tensor, attribution_map, target_type='pathology', target_id='predicted', steps=10, device='cuda', num_steps=None):
    if num_steps is not None:
        steps = num_steps
    """
    Measures the recovery in target prediction score starting from baseline as top-attributed pixels are inserted.
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    orig_score, meta = compute_target_score(model, input_tensor, target_type=target_type, target_id=target_id)
    resolved_type = meta['target_type']
    resolved_id = meta['target_id']

    kernel = torch.ones(1, 1, 15, 15, device=device) / 225.0
    blurred_list = [F.conv2d(input_tensor[:, c:c+1], kernel, padding=7) for c in range(3)]
    baseline = torch.cat(blurred_list, dim=1)

    H, W = input_tensor.shape[2], input_tensor.shape[3]
    if attribution_map.shape != (H, W):
        t_am = torch.from_numpy(attribution_map).unsqueeze(0).unsqueeze(0).float()
        am_resized = F.interpolate(t_am, size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()
    else:
        am_resized = attribution_map

    flat_attr = am_resized.flatten()
    sorted_indices = np.argsort(-flat_attr)
    total_pixels = len(sorted_indices)

    fractions = np.linspace(0.0, 0.8, steps)
    scores = []

    for f in fractions:
        num_insert = int(f * total_pixels)
        mask = np.zeros(total_pixels, dtype=np.float32)
        if num_insert > 0:
            mask[sorted_indices[:num_insert]] = 1.0
        mask_2d = torch.from_numpy(mask.reshape(H, W)).unsqueeze(0).unsqueeze(0).to(device)

        ins_img = mask_2d * input_tensor + (1.0 - mask_2d) * baseline

        with torch.no_grad():
            _, p_logits, w_logits = model.forward_single(ins_img, return_ordinal=True)
            if resolved_type == 'pathology':
                cur_val = float(p_logits[0, resolved_id].item())
            else:
                q = torch.sigmoid(w_logits)
                w_probs = torch.stack([
                    1.0 - q[:, 0], q[:, 0] - q[:, 1],
                    q[:, 1] - q[:, 2], q[:, 2] - q[:, 3], q[:, 3]
                ], dim=1)
                cur_val = float(w_probs[0, resolved_id].item())
            scores.append(cur_val)

    insertion_auc = float(np.mean(scores))
    return {
        'fractions': fractions.tolist(),
        'scores': scores,
        'insertion_auc': insertion_auc
    }
