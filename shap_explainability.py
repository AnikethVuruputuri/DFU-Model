"""
Comprehensive SHAP Explainability Analysis for DFU-SIAM Cross-Taxonomy Fusion Model.
Generates the FULL suite of SHAP plots & statistical graphs:

  1. Image saliency overlays (pathology + depth)
  2. Mean Absolute SHAP Attribution Bar Chart (per class)
  3. Per-Class SHAP Attribution Distribution (violin/box plots)
  4. Channel-wise (R/G/B) SHAP Contribution Analysis
  5. Spatial Attention Heatmap (averaged across samples)
  6. Prediction Confidence vs SHAP Magnitude Scatter Plot
  7. SHAP Waterfall-style Bar Plot (per sample, per class)
  8. Embedding Feature SHAP Beeswarm / Summary Plot (CNN vs ViT)
  9. Cross-Head Attribution Correlation (Pathology vs Depth)
 10. Feature Branch Importance Bar Chart
 11. Joint UT-Grid Comparative Saliency
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F

from dfu_siam_model import DFUSiamNet
from dataset import DFUSingleDataset, WagnerSingleDataset, get_dfu_transforms


# ──────────────────────── Model Wrappers ────────────────────────

class DFUPathologyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        emb = self.model.backbone(x)
        logits = self.model.classifier(emb)
        return F.softmax(logits, dim=1)


class WagnerDepthWrapper(nn.Module):
    def __init__(self, model, anchor):
        super().__init__()
        self.model = model
        self.register_buffer('anchor', anchor)

    def forward(self, x):
        emb = self.model.backbone(x)
        proj = self.model.ordinal_head(emb)
        return torch.norm(proj - self.anchor, p=2, dim=1, keepdim=True)


# ──────────────────────── Core Attribution ────────────────────────

def compute_integrated_gradients(model_fn, inp, baseline, target_class=None, steps=20):
    """Path-Integrated Gradients (Aumann-Shapley attribution)."""
    inp = inp.clone().detach()
    baseline = baseline.clone().detach()
    diff = inp - baseline
    total_grads = torch.zeros_like(inp)

    for alpha in np.linspace(0.0, 1.0, steps):
        interp = baseline + float(alpha) * diff
        interp.requires_grad_(True)
        out = model_fn(interp)
        score = out[:, target_class] if target_class is not None else out.squeeze()
        score.backward()
        total_grads += interp.grad.detach()
        del interp, out, score
        if inp.is_cuda:
            torch.cuda.empty_cache()

    return (diff * total_grads / steps).detach().cpu().numpy()


# ──────────────────────── Utilities ────────────────────────

def load_model(ckpt_path, device):
    model = DFUSiamNet(num_classes=4, num_wagner_classes=5,
                       ordinal_type='anchor_metric', dropout=0.0)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()

    anchor = torch.zeros(1, 128, device=device)
    for k in ['ordinal_criterion_state_dict', 'ordinal_loss_state_dict']:
        if k in ckpt and 'anchor' in ckpt[k]:
            anchor = ckpt[k]['anchor'].to(device)
            print(f"[SHAP] Successfully loaded learned anchor vector from '{k}'")
            break
    else:
        print("[SHAP] Warning: No learned anchor found in checkpoint; defaulting to zeros.")
    return model, anchor


def denorm(t):
    m = np.array([.485, .456, .406]).reshape(1, 1, 3)
    s = np.array([.229, .224, .225]).reshape(1, 1, 3)
    return np.clip(s * t.cpu().numpy().transpose(1, 2, 0) + m, 0, 1)


CLASS_NAMES = ['None', 'Infection', 'Ischaemia', 'Both']
CLASS_COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#9467bd']
CHANNEL_NAMES = ['Red', 'Green', 'Blue']
CHANNEL_COLORS = ['#e74c3c', '#27ae60', '#2980b9']


# ──────────────────────── Plot Functions ────────────────────────

def plot_01_saliency_overlays(shap_path, shap_depth, test_t, labels, out):
    """Plot 1: DFU Pathology + Depth saliency overlays."""
    N = len(labels)
    fig, axes = plt.subplots(N, 6, figsize=(22, 4 * N))
    if N == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(N):
        img = denorm(test_t[i])
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"Image #{i+1}\nTrue: {CLASS_NAMES[labels[i]]}", fontweight='bold'); axes[i, 0].axis('off')
        for c in range(4):
            a = np.sum(np.abs(shap_path[c][i]), axis=0)
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            axes[i, c+1].imshow(img); axes[i, c+1].imshow(a, cmap='jet', alpha=0.55)
            axes[i, c+1].set_title(f"SHAP: {CLASS_NAMES[c]}"); axes[i, c+1].axis('off')
        d = np.sum(np.abs(shap_depth[i]), axis=0)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        axes[i, 5].imshow(img); axes[i, 5].imshow(d, cmap='hot', alpha=0.55)
        axes[i, 5].set_title("SHAP: Depth"); axes[i, 5].axis('off')

    plt.suptitle("SHAP Pixel Attribution Overlays — Pathology Classes & Depth Metric", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(out, '01_shap_saliency_overlays.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_02_mean_abs_bar(shap_path, shap_depth, out):
    """Plot 2: Mean |SHAP| attribution magnitude per class (bar chart)."""
    means = [np.mean(np.abs(np.array(shap_path[c]))) for c in range(4)]
    depth_mean = np.mean(np.abs(np.array(shap_depth)))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = CLASS_NAMES + ['Depth Metric']
    vals = means + [depth_mean]
    colors = CLASS_COLORS + ['#e67e22']
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=1.0, width=0.55)
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.6f}', xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('Mean |SHAP| Attribution', fontsize=12)
    ax.set_title('Mean Absolute SHAP Attribution per Output Head', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(out, '02_shap_mean_abs_bar.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_03_violin_box(shap_path, out):
    """Plot 3: Per-class SHAP attribution distribution (violin + box)."""
    data = [np.abs(np.array(shap_path[c])).flatten() for c in range(4)]
    # Subsample for efficiency
    data_sub = [d[np.random.choice(len(d), min(5000, len(d)), replace=False)] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Violin
    vp = ax1.violinplot(data_sub, showmeans=True, showmedians=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(CLASS_COLORS[i]); body.set_alpha(0.6)
    ax1.set_xticks(range(1, 5)); ax1.set_xticklabels(CLASS_NAMES)
    ax1.set_ylabel('|SHAP| Value'); ax1.set_title('SHAP Attribution Distribution (Violin)', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Box
    bp = ax2.boxplot(data_sub, patch_artist=True, notch=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CLASS_COLORS[i]); patch.set_alpha(0.6)
    ax2.set_xticks(range(1, 5)); ax2.set_xticklabels(CLASS_NAMES)
    ax2.set_ylabel('|SHAP| Value'); ax2.set_title('SHAP Attribution Distribution (Box Plot)', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle('Per-Class SHAP Attribution Spread Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '03_shap_violin_box.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_04_channel_analysis(shap_path, shap_depth, out):
    """Plot 4: Channel-wise (R/G/B) mean |SHAP| contribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pathology head — channel breakdown per class
    x = np.arange(4)
    w = 0.22
    for ch in range(3):
        vals = [np.mean(np.abs(np.array(shap_path[c])[:, ch, :, :])) for c in range(4)]
        axes[0].bar(x + ch * w, vals, width=w, label=CHANNEL_NAMES[ch],
                    color=CHANNEL_COLORS[ch], edgecolor='black', linewidth=0.8)
    axes[0].set_xticks(x + w); axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_ylabel('Mean |SHAP|'); axes[0].set_title('Pathology Head: Channel-wise Attribution', fontweight='bold')
    axes[0].legend(); axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Depth head — channel breakdown
    depth_arr = np.array(shap_depth)
    ch_vals = [np.mean(np.abs(depth_arr[:, ch, :, :])) for ch in range(3)]
    bars = axes[1].bar(CHANNEL_NAMES, ch_vals, color=CHANNEL_COLORS, edgecolor='black', linewidth=0.8, width=0.45)
    for b in bars:
        h = b.get_height()
        axes[1].annotate(f'{h:.6f}', xy=(b.get_x() + b.get_width()/2, h),
                         xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')
    axes[1].set_ylabel('Mean |SHAP|'); axes[1].set_title('Depth Head: Channel-wise Attribution', fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle('RGB Channel-wise SHAP Contribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '04_shap_channel_analysis.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_05_spatial_average_heatmap(shap_path, shap_depth, out):
    """Plot 5: Mean spatial attribution heatmap averaged across all samples."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    titles = CLASS_NAMES + ['Depth Metric']
    cmaps = ['Reds', 'Oranges', 'Blues', 'Purples', 'YlOrBr']

    for c in range(4):
        avg = np.mean([np.sum(np.abs(shap_path[c][i]), axis=0) for i in range(len(shap_path[0]))], axis=0)
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
        im = axes[c].imshow(avg, cmap=cmaps[c], interpolation='bilinear')
        axes[c].set_title(f"Mean SHAP: {titles[c]}", fontweight='bold'); axes[c].axis('off')
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    avg_d = np.mean([np.sum(np.abs(shap_depth[i]), axis=0) for i in range(len(shap_depth))], axis=0)
    avg_d = (avg_d - avg_d.min()) / (avg_d.max() - avg_d.min() + 1e-8)
    im = axes[4].imshow(avg_d, cmap=cmaps[4], interpolation='bilinear')
    axes[4].set_title(f"Mean SHAP: {titles[4]}", fontweight='bold'); axes[4].axis('off')
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.suptitle('Spatially-Averaged SHAP Attribution Heatmaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '05_shap_spatial_average_heatmap.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_06_confidence_scatter(shap_path, shap_depth, pred_probs, pred_depths, out):
    """Plot 6: Prediction confidence vs SHAP magnitude scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pathology: max prob vs total |SHAP|
    max_probs = np.max(pred_probs, axis=1)
    total_shap = [sum(np.sum(np.abs(shap_path[c][i])) for c in range(4)) for i in range(len(max_probs))]
    pred_classes = np.argmax(pred_probs, axis=1)

    for i in range(len(max_probs)):
        ax1.scatter(max_probs[i], total_shap[i], c=CLASS_COLORS[pred_classes[i]],
                    s=120, edgecolor='black', linewidth=1.2, zorder=5)
    ax1.set_xlabel('Prediction Confidence (max softmax)', fontsize=11)
    ax1.set_ylabel('Total |SHAP| Attribution', fontsize=11)
    ax1.set_title('Pathology: Confidence vs SHAP Magnitude', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Add legend
    for c in range(4):
        ax1.scatter([], [], c=CLASS_COLORS[c], label=CLASS_NAMES[c], s=80, edgecolor='black')
    ax1.legend(fontsize=9)

    # Depth: predicted depth vs |SHAP|
    depth_shap = [np.sum(np.abs(shap_depth[i])) for i in range(len(pred_depths))]
    ax2.scatter(pred_depths, depth_shap, c='#e67e22', s=120, edgecolor='black', linewidth=1.2, zorder=5)
    for i in range(len(pred_depths)):
        ax2.annotate(f'#{i+1}', (pred_depths[i], depth_shap[i]), textcoords='offset points',
                     xytext=(5, 5), fontsize=9, fontweight='bold')
    ax2.set_xlabel('Predicted Depth Score', fontsize=11)
    ax2.set_ylabel('Total |SHAP| Attribution', fontsize=11)
    ax2.set_title('Depth: Predicted Score vs SHAP Magnitude', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Model Confidence vs Explainability Magnitude', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '06_shap_confidence_scatter.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_07_waterfall(shap_path, pred_probs, labels, out):
    """Plot 7: SHAP waterfall-style horizontal bar for each sample."""
    N = len(labels)
    fig, axes = plt.subplots(N, 1, figsize=(10, 3.5 * N))
    if N == 1:
        axes = [axes]

    for i in range(N):
        class_shaps = [np.sum(np.abs(shap_path[c][i])) for c in range(4)]
        total = sum(class_shaps)
        fracs = [s / (total + 1e-8) for s in class_shaps]
        pred_cls = np.argmax(pred_probs[i])

        bars = axes[i].barh(CLASS_NAMES, fracs, color=CLASS_COLORS, edgecolor='black', linewidth=0.8)
        for j, b in enumerate(bars):
            axes[i].text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                        f'{fracs[j]*100:.1f}%', va='center', fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Fraction of Total |SHAP|')
        axes[i].set_title(f'Sample #{i+1} — True: {CLASS_NAMES[labels[i]]}, Predicted: {CLASS_NAMES[pred_cls]} ({pred_probs[i][pred_cls]*100:.1f}%)',
                         fontweight='bold', fontsize=11)
        axes[i].set_xlim(0, max(fracs) * 1.25)
        axes[i].grid(axis='x', linestyle='--', alpha=0.3)

    plt.suptitle('SHAP Waterfall: Per-Sample Class Attribution Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '07_shap_waterfall.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_08_embedding_beeswarm(emb_shap_cnn, emb_shap_vit, out):
    """Plot 8: Embedding-level feature SHAP summary (beeswarm-style)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CNN branch top-k features
    cnn_mean = np.mean(np.abs(emb_shap_cnn), axis=0)
    topk = 20
    top_idx = np.argsort(cnn_mean)[-topk:][::-1]
    axes[0].barh(range(topk), cnn_mean[top_idx][::-1], color='#1f77b4', edgecolor='black', linewidth=0.6)
    axes[0].set_yticks(range(topk))
    axes[0].set_yticklabels([f'CNN feat {j}' for j in top_idx[::-1]], fontsize=8)
    axes[0].set_xlabel('Mean |SHAP| Attribution')
    axes[0].set_title('EfficientNetV2-S: Top-20 Feature Attributions', fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.3)

    # ViT branch top-k features
    vit_mean = np.mean(np.abs(emb_shap_vit), axis=0)
    top_idx_v = np.argsort(vit_mean)[-topk:][::-1]
    axes[1].barh(range(topk), vit_mean[top_idx_v][::-1], color='#ff7f0e', edgecolor='black', linewidth=0.6)
    axes[1].set_yticks(range(topk))
    axes[1].set_yticklabels([f'ViT feat {j}' for j in top_idx_v[::-1]], fontsize=8)
    axes[1].set_xlabel('Mean |SHAP| Attribution')
    axes[1].set_title('Swin-Tiny: Top-20 Feature Attributions', fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.3)

    plt.suptitle('Embedding-Level SHAP Feature Importance (Beeswarm Summary)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '08_shap_embedding_beeswarm.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_09_cross_head_correlation(shap_path, shap_depth, out):
    """Plot 9: Cross-head attribution correlation (pathology vs depth)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    depth_flat = np.array([np.sum(np.abs(shap_depth[i]), axis=0).flatten() for i in range(len(shap_depth))])

    for c in range(4):
        path_flat = np.array([np.sum(np.abs(shap_path[c][i]), axis=0).flatten() for i in range(len(shap_path[0]))])
        # Subsample for scatter
        n_pts = min(2000, path_flat.shape[1])
        idx = np.random.choice(path_flat.shape[1], n_pts, replace=False)

        for i in range(len(shap_path[0])):
            axes[c].scatter(path_flat[i, idx], depth_flat[i, idx], alpha=0.2, s=5, c=CLASS_COLORS[c])

        corr = np.corrcoef(path_flat.flatten(), depth_flat.flatten())[0, 1]
        axes[c].set_xlabel(f'{CLASS_NAMES[c]} |SHAP|', fontsize=10)
        axes[c].set_ylabel('Depth |SHAP|', fontsize=10)
        axes[c].set_title(f'{CLASS_NAMES[c]} vs Depth\nr = {corr:.3f}', fontweight='bold')
        axes[c].grid(True, linestyle='--', alpha=0.2)

    plt.suptitle('Cross-Head Pixel Attribution Correlation: Pathology vs Depth', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '09_shap_cross_head_correlation.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_10_feature_branch(embs, out):
    """Plot 10: Feature branch importance bar chart."""
    cnn_imp = float(np.mean(np.abs(embs[:, :1280])))
    vit_imp = float(np.mean(np.abs(embs[:, 1280:])))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['EfficientNetV2-S\n(CNN Branch)', 'Swin-Tiny\n(ViT Branch)'],
                  [cnn_imp, vit_imp], color=['#1f77b4', '#ff7f0e'],
                  width=0.45, edgecolor='black', linewidth=1.2)
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Embedding Magnitude', fontsize=12)
    ax.set_title('Cross-Taxonomy Feature Contribution per Backbone Branch', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(cnn_imp, vit_imp) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(out, '10_shap_feature_branch.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def plot_11_ut_grid_saliency(shap_path, shap_depth, test_t, out):
    """Plot 11: Joint UT-Grid saliency comparison."""
    N = len(shap_depth)
    fig, axes = plt.subplots(2, N, figsize=(4.5 * N, 7.5))

    for i in range(N):
        img = denorm(test_t[i])
        a_p = np.sum(np.abs(shap_path[1][i]), axis=0)
        a_p = (a_p - a_p.min()) / (a_p.max() - a_p.min() + 1e-8)
        a_d = np.sum(np.abs(shap_depth[i]), axis=0)
        a_d = (a_d - a_d.min()) / (a_d.max() - a_d.min() + 1e-8)

        axes[0, i].imshow(img); axes[0, i].imshow(a_p, cmap='viridis', alpha=0.5)
        axes[0, i].set_title(f'Sample #{i+1}: Infection', fontweight='bold'); axes[0, i].axis('off')
        axes[1, i].imshow(img); axes[1, i].imshow(a_d, cmap='magma', alpha=0.5)
        axes[1, i].set_title(f'Sample #{i+1}: Depth', fontweight='bold'); axes[1, i].axis('off')

    plt.suptitle('Joint UT-Grid Saliency: Infection Stage vs Depth Metric', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, '11_shap_ut_grid_saliency.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


# ──────────────────────── Main Pipeline ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full SHAP Analysis Suite for DFU-SIAM")
    parser.add_argument('--checkpoint', default='checkpoints/dfu_siam_best.pth')
    parser.add_argument('--num_test', type=int, default=4)
    parser.add_argument('--output_dir', default='output_shap')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"[SHAP] Device: {device} ({gpu_name})")

    model, anchor = load_model(args.checkpoint, device)
    transform = get_dfu_transforms(is_train=False)
    dfu_ds = DFUSingleDataset(
        os.path.join('DFUdataset', 'DFUC2021_train', 'images'),
        csv_file=os.path.join('DFUdataset', 'DFUC2021_train', 'train.csv'),
        transform=transform
    )

    test_tensors, test_labels = [], []
    for i in range(args.num_test):
        img, lbl, _ = dfu_ds[i + 50]
        test_tensors.append(img.unsqueeze(0))
        test_labels.append(lbl)
    test_tensors = torch.cat(test_tensors, dim=0).to(device)
    baseline = torch.zeros(1, 3, 224, 224, device=device)

    # ─── Compute pathology SHAP ───
    print("\n=== Phase 1: Computing Pathology SHAP Attributions ===")
    wrap_path = DFUPathologyWrapper(model).to(device).eval()
    shap_path = [[] for _ in range(4)]
    for c in range(4):
        print(f"  Class [{c}] {CLASS_NAMES[c]}...")
        for i in range(args.num_test):
            attr = compute_integrated_gradients(wrap_path, test_tensors[i:i+1], baseline, target_class=c, steps=args.steps)
            shap_path[c].append(attr[0])

    # ─── Compute depth SHAP ───
    print("\n=== Phase 2: Computing Depth SHAP Attributions ===")
    wrap_depth = WagnerDepthWrapper(model, anchor).to(device).eval()
    shap_depth = []
    for i in range(args.num_test):
        print(f"  Sample #{i+1}...")
        attr = compute_integrated_gradients(wrap_depth, test_tensors[i:i+1], baseline, target_class=None, steps=args.steps)
        shap_depth.append(attr[0])

    # ─── Get model predictions ───
    print("\n=== Phase 3: Collecting Model Predictions ===")
    with torch.no_grad():
        embs = model.backbone(test_tensors)
        logits = model.classifier(embs)
        pred_probs = F.softmax(logits, dim=1).cpu().numpy()

        ord_proj = model.ordinal_head(embs)
        pred_depths = torch.norm(ord_proj - anchor, p=2, dim=1).cpu().numpy()

        embs_np = embs.cpu().numpy()

    # ─── Compute embedding-level SHAP (via gradient to classifier) ───
    print("\n=== Phase 4: Computing Embedding-Level Feature Attributions ===")
    emb_shap_all = []
    for i in range(args.num_test):
        inp = test_tensors[i:i+1].clone().detach().requires_grad_(True)
        emb = model.backbone(inp)
        emb.retain_grad()
        pred_class = np.argmax(pred_probs[i])
        logit_val = model.classifier(emb)[:, pred_class]
        logit_val.backward()
        grad_emb = emb.grad.detach().cpu().numpy()[0]
        emb_shap_all.append(grad_emb * embs_np[i])
        del inp, emb, logit_val
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    emb_shap_all = np.array(emb_shap_all)
    emb_shap_cnn = emb_shap_all[:, :1280]
    emb_shap_vit = emb_shap_all[:, 1280:]

    # ─── Generate all plots ───
    print("\n=== Phase 5: Generating All SHAP Plots & Graphs ===")

    print("\n [1/11] Saliency Overlays...")
    plot_01_saliency_overlays(shap_path, shap_depth, test_tensors, test_labels, args.output_dir)

    print(" [2/11] Mean Absolute SHAP Bar Chart...")
    plot_02_mean_abs_bar(shap_path, shap_depth, args.output_dir)

    print(" [3/11] Attribution Distribution (Violin + Box)...")
    plot_03_violin_box(shap_path, args.output_dir)

    print(" [4/11] Channel-wise (RGB) Analysis...")
    plot_04_channel_analysis(shap_path, shap_depth, args.output_dir)

    print(" [5/11] Spatial Average Heatmaps...")
    plot_05_spatial_average_heatmap(shap_path, shap_depth, args.output_dir)

    print(" [6/11] Confidence vs SHAP Scatter...")
    plot_06_confidence_scatter(shap_path, shap_depth, pred_probs, pred_depths, args.output_dir)

    print(" [7/11] Waterfall Breakdown...")
    plot_07_waterfall(shap_path, pred_probs, test_labels, args.output_dir)

    print(" [8/11] Embedding Beeswarm Summary...")
    plot_08_embedding_beeswarm(emb_shap_cnn, emb_shap_vit, args.output_dir)

    print(" [9/11] Cross-Head Correlation...")
    plot_09_cross_head_correlation(shap_path, shap_depth, args.output_dir)

    print(" [10/11] Feature Branch Importance...")
    plot_10_feature_branch(embs_np, args.output_dir)

    print(" [11/11] UT-Grid Joint Saliency...")
    plot_11_ut_grid_saliency(shap_path, shap_depth, test_tensors, args.output_dir)

    print(f"\n [SHAP COMPLETE] All 11 explainability plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
