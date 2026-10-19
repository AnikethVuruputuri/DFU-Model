"""
Evaluation and Cross-Taxonomy Metrics for DFU-SIAM v3 Architecture.
Provides:
1. evaluate_wagner_ordinal: Computes Accuracy, MAE, Adjacent Accuracy, and Loss on held-out Wagner validation set.
2. predict_ut_grid: Probabilistic synthesis of exploratory 16-cell UT wound matrix from independent predictions.
   NOTE: This is an exploratory probabilistic visualization tool, NOT a ground-truth paired 16-cell classifier.
3. evaluate_zero_shot_cross_taxonomy: Evaluates cross-taxonomy Spearman rank correlation and depth deltas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr


def evaluate_wagner_ordinal(model, dataloader, device, ordinal_loss_fn):
    """
    Evaluates Wagner depth prediction performance on held-out Wagner validation dataset.
    Returns: accuracy, MAE, adjacent_accuracy (within +-1 grade), mean_loss
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, lbls, _ in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            out = model.forward_wagner(imgs)
            # Handle both v3 (3 items: z_shared, z_depth, wagner_logits) and legacy (2 items: emb, ord_out)
            ord_out = out[-1]

            loss, logits, preds = ordinal_loss_fn(ord_out, lbls)

            total_loss += loss.item()
            num_batches += 1
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    adj_acc = np.mean(np.abs(all_preds - all_targets) <= 1)
    mean_loss = total_loss / max(1, num_batches)
    
    from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(5)))
    per_grade_f1 = f1_score(all_targets, all_preds, average=None, labels=list(range(5)), zero_division=0)
    grade_names = [f"Grade {g}" for g in range(5)]

    return {
        'accuracy': float(acc),
        'mae': float(mae),
        'adjacent_accuracy': float(adj_acc),
        'macro_f1': float(macro_f1),
        'quadratic_weighted_kappa': float(qwk),
        'loss': float(mean_loss),
        'per_grade_f1': {g: float(f) for g, f in zip(grade_names, per_grade_f1)},
        'confusion_matrix': cm.tolist()
    }


def predict_ut_grid(dfu_probs, wagner_out, scale=1.0):
    """
    Combines DFU pathology probabilities (4,) and Wagner ordinal output into an exploratory 16-cell UT wound grid (4 x 4).
    UT Depth: 0 (Wagner 0), 1 (Wagner 1), 2 (Wagner 2), 3 (Wagner 3 & 4)
    UT Stage: Stage A (None), Stage B (Infection), Stage C (Ischaemia), Stage D (Both)

    IMPORTANT RESEARCH NOTE:
    This tensor product (p_depth \otimes p_pathology) is an exploratory probabilistic synthesis.
    It does not represent a clinically validated paired 16-cell ground-truth classifier, as DFUC2021
    and Wagner cohorts originate from disjoint clinical cohorts with distinct annotation taxonomies.
    """
    # CORAL cumulative logit decoding
    wagner_arr = np.asarray(wagner_out)
    if wagner_arr.ndim > 1:
        wagner_arr = wagner_arr.squeeze()

    cum_probs = 1.0 / (1.0 + np.exp(-wagner_arr[:4]))  # Sigmoid on 4 cutoff logits
    p0 = 1.0 - cum_probs[0]
    p1 = cum_probs[0] - cum_probs[1]
    p2 = cum_probs[1] - cum_probs[2]
    p3 = cum_probs[2]  # Combined Grade 3 & Grade 4 for 4-depth UT standard
    depth_probs = np.array([p0, p1, p2, p3])
    depth_probs = np.maximum(depth_probs, 0.0)
    depth_probs = depth_probs / (np.sum(depth_probs) + 1e-8)

    ut_grid = np.outer(depth_probs, dfu_probs)
    return ut_grid, depth_probs


def evaluate_zero_shot_cross_taxonomy(model, dfu_val_loader, wagner_val_loader, device, ordinal_loss_fn):
    """
    Zero-Shot Cross-Taxonomy Consistency Evaluation:
    1. Evaluates DFUC categorical pathology head on held-out Wagner images (never trained on DFUC labels).
       Checks Spearman correlation between Wagner ground-truth depth (0..4) and predicted pathology risk.
    2. Evaluates Wagner ordinal depth head on held-out DFUC images (never trained on Wagner labels).
       Checks mean predicted depth per DFUC pathology class (none, infection, ischaemia, both).
    """
    model.eval()

    # Part 1: Categorical Head on Wagner Validation Set
    wagner_true_grades = []
    predicted_pathology_risks = []

    with torch.no_grad():
        for imgs, lbls, _ in wagner_val_loader:
            imgs = imgs.to(device)
            out = model.forward_single(imgs)
            dfu_logits = out[1] if isinstance(out, tuple) else out
            dfu_probs = F.softmax(dfu_logits, dim=1).cpu().numpy()

            pathology_risk = 1.0 - dfu_probs[:, 0]
            predicted_pathology_risks.extend(pathology_risk)
            wagner_true_grades.extend(lbls.numpy())

    wagner_true_grades = np.array(wagner_true_grades)
    predicted_pathology_risks = np.array(predicted_pathology_risks)

    spearman_rho, spearman_p = spearmanr(wagner_true_grades, predicted_pathology_risks)
    pearson_r, pearson_p = pearsonr(wagner_true_grades, predicted_pathology_risks)

    # Part 2: Ordinal Depth Head on DFUC Validation Set
    dfu_true_classes = []
    predicted_wagner_depths = []

    with torch.no_grad():
        for imgs, lbls, _ in dfu_val_loader:
            imgs = imgs.to(device)
            out = model.forward_wagner(imgs)
            ord_out = out[-1]

            _, logits, preds = ordinal_loss_fn(ord_out, lbls.to(device))

            predicted_wagner_depths.extend(preds.cpu().numpy())
            dfu_true_classes.extend(lbls.numpy())

    dfu_true_classes = np.array(dfu_true_classes)
    predicted_wagner_depths = np.array(predicted_wagner_depths)

    mean_depth_per_dfu_class = {}
    class_names = ['none', 'infection', 'ischaemia', 'both']
    for c_idx, c_name in enumerate(class_names):
        mask = (dfu_true_classes == c_idx)
        if np.sum(mask) > 0:
            mean_depth_per_dfu_class[c_name] = float(np.mean(predicted_wagner_depths[mask]))
        else:
            mean_depth_per_dfu_class[c_name] = 0.0

    pathological_mask = (dfu_true_classes > 0)
    clean_mask = (dfu_true_classes == 0)
    pathological_mean_depth = float(np.mean(predicted_wagner_depths[pathological_mask])) if np.sum(pathological_mask) > 0 else 0.0
    clean_mean_depth = float(np.mean(predicted_wagner_depths[clean_mask])) if np.sum(clean_mask) > 0 else 0.0

    return {
        'wagner_spearman_rho': float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
        'wagner_spearman_p': float(spearman_p) if not np.isnan(spearman_p) else 1.0,
        'wagner_pearson_r': float(pearson_r) if not np.isnan(pearson_r) else 0.0,
        'mean_depth_per_dfu_class': mean_depth_per_dfu_class,
        'pathological_vs_clean_depth_delta': float(pathological_mean_depth - clean_mean_depth)
    }
