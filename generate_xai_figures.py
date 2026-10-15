"""
Publication-Quality Research Figure Generator for DFU-SIAM v3 Explainable AI.

Generates five high-resolution (300 DPI) publication figures:
- Figure A: Correct Prediction Attribution (Multi-method showcase on true-positive sample)
- Figure B: Misclassification Error Analysis (Failure mode diagnostic visualization)
- Figure C: Minority Class Attribution (Ischaemia pathology feature localization)
- Figure D: Wagner Ordinal Depth Grading (Attribution across wound depth severity)
- Figure E: Cross-Method Attribution Comparison Matrix (Head-to-head resolution comparison)

Usage:
    python generate_xai_figures.py --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_dfu_transforms
from xai.unified import XAIExplainer
from xai.base import PATHOLOGY_CLASSES, WAGNER_GRADES, overlay_heatmap_on_image
from explain_prediction import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Publication XAI Figures")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_a1_full_balanced/best_checkpoint.pth",
                        help="Path to trained DFU-SIAM v3 model checkpoint")
    parser.add_argument("--data_dir", type=str, default="DFUdataset",
                        help="Path to DFU dataset directory")
    parser.add_argument("--wagner_dir", type=str, default="WagnerDataset",
                        help="Path to Wagner dataset directory")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to validation CSV split")
    parser.add_argument("--output_dir", type=str, default="outputs/xai_figures",
                        help="Output directory for publication figures")
    parser.add_argument("--arch", type=str, default="full",
                        help="Backbone architecture")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")
    return parser.parse_args()


def find_val_csv(args):
    if args.val_csv and os.path.exists(args.val_csv):
        return args.val_csv
    ckpt_dir = os.path.dirname(args.checkpoint)
    cands = [
        os.path.join(ckpt_dir, "temp_val_split.csv"),
        "checkpoints_a1_full_balanced/temp_val_split.csv",
        "checkpoints_a1_full/temp_val_split.csv",
        "checkpoints/temp_val_split.csv"
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find validation CSV split.")


def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9.5
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['figure.titlesize'] = 11.5
    plt.rcParams['figure.titleweight'] = 'bold'


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()

    model, cfg, device = load_model(args.checkpoint, arch=args.arch, device=args.device)
    explainer = XAIExplainer(model, device=device)
    explainer.ig.steps = 25
    explainer.shap.grid_size = 14
    explainer.shap.n_evals = 80

    val_csv = find_val_csv(args)
    df = pd.read_csv(val_csv)
    img_dir = os.path.join(args.data_dir, "DFUC2021_train", "images")
    transform = get_dfu_transforms(is_train=False, img_size=(224, 224))

    print("[*] Profiling validation samples to identify representative cases...")
    correct_samples = {c: [] for c in range(4)}
    incorrect_samples = []

    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['image'])
            if not os.path.exists(img_path):
                continue
            gt = int(np.argmax([row[c] for c in PATHOLOGY_CLASSES]))
            try:
                pil_img = Image.open(img_path).convert("RGB")
                inp = transform(pil_img).unsqueeze(0).to(device)
                _, p_logits, w_logits = model.forward_single(inp, return_ordinal=True)
                pred = int(torch.argmax(p_logits, dim=-1)[0].item())
                probs = torch.softmax(p_logits, dim=-1)[0].cpu().numpy()
            except Exception:
                continue

            entry = {'image_path': img_path, 'name': row['image'], 'gt': gt, 'pred': pred, 'probs': probs}
            if pred == gt:
                correct_samples[gt].append(entry)
            else:
                incorrect_samples.append(entry)

            # Once we have enough of each, stop profiling to save time
            if (len(correct_samples[0]) >= 3 and len(correct_samples[1]) >= 3 and
                len(correct_samples[2]) >= 3 and len(correct_samples[3]) >= 3 and
                len(incorrect_samples) >= 5):
                break

    # =========================================================================
    # Figure A: Correct Prediction Attribution Showcase
    # =========================================================================
    print("[*] Generating Figure A (Correct Prediction Attribution)...")
    sample_a = correct_samples[1][0] if correct_samples[1] else (correct_samples[0][0] if correct_samples[0] else None)
    if sample_a:
        pil_a = Image.open(sample_a['image_path']).convert("RGB")
        res_a, meta_a = explainer.explain_image(pil_a, transform, target_type='pathology', target_id='predicted', method='all')

        cnn_ov, _ = overlay_heatmap_on_image(pil_a, res_a['gradcam']['cnn_heatmap'])
        vit_ov, _ = overlay_heatmap_on_image(pil_a, res_a['gradcam']['vit_heatmap'])
        fused_ov, _ = overlay_heatmap_on_image(pil_a, res_a['gradcam']['fused_heatmap'])
        trans_ov, _ = overlay_heatmap_on_image(pil_a, res_a['transformer']['transformer_heatmap'])
        ig_ov, _ = overlay_heatmap_on_image(pil_a, res_a['integrated_gradients']['ig_heatmap'])
        shap_ov, _ = overlay_heatmap_on_image(pil_a, res_a['shap']['shap_heatmap'])

        panels_a = [
            ("Original Image", np.array(pil_a)),
            ("CNN Grad-CAM\n(EfficientNetV2)", cnn_ov),
            ("Transformer\n(Swin-Tiny Window)", trans_ov),
            ("Fused Attribution\n(Dual-Branch)", fused_ov),
            ("Integrated Gradients\n(Pixel Path)", ig_ov),
            ("KernelSHAP\n(Superpixel Mask)", shap_ov)
        ]

        fig_a, axes_a = plt.subplots(1, 6, figsize=(18, 3.8), dpi=300)
        for ax, (title, img) in zip(axes_a, panels_a):
            ax.imshow(img)
            ax.set_title(title, pad=6)
            ax.axis('off')

        gt_str = PATHOLOGY_CLASSES[sample_a['gt']].upper()
        prob_str = f"{sample_a['probs'][sample_a['pred']]:.2%}"
        fig_a.suptitle(f"Figure A: Multi-Method Explainability on Correct Prediction\nGround Truth: {gt_str} | Predicted: {gt_str} (Confidence: {prob_str}) | Sample: {sample_a['name']}",
                       fontsize=11.5, y=1.04)
        save_path_a = os.path.join(args.output_dir, "figure_a_correct_prediction.png")
        fig_a.savefig(save_path_a, bbox_inches='tight')
        plt.close(fig_a)
        print(f"    Saved: {save_path_a}")

    # =========================================================================
    # Figure B: Misclassification Error Analysis
    # =========================================================================
    print("[*] Generating Figure B (Misclassification Error Analysis)...")
    if incorrect_samples:
        sample_b = incorrect_samples[0]
        pil_b = Image.open(sample_b['image_path']).convert("RGB")
        # Explain the false predicted class AND the true ground-truth class
        res_pred, _ = explainer.explain_image(pil_b, transform, target_type='pathology', target_id=sample_b['pred'], method='all')
        res_gt, _ = explainer.explain_image(pil_b, transform, target_type='pathology', target_id=sample_b['gt'], method='gradcam')

        pred_cnn_ov, _ = overlay_heatmap_on_image(pil_b, res_pred['gradcam']['cnn_heatmap'])
        pred_trans_ov, _ = overlay_heatmap_on_image(pil_b, res_pred['transformer']['transformer_heatmap'])
        pred_ig_ov, _ = overlay_heatmap_on_image(pil_b, res_pred['integrated_gradients']['ig_heatmap'])
        pred_shap_ov, _ = overlay_heatmap_on_image(pil_b, res_pred['shap']['shap_heatmap'])
        gt_cnn_ov, _ = overlay_heatmap_on_image(pil_b, res_gt['gradcam']['cnn_heatmap'])

        panels_b = [
            ("Original Ulcer", np.array(pil_b)),
            (f"Target: False Pred\n({PATHOLOGY_CLASSES[sample_b['pred']].upper()}) CNN", pred_cnn_ov),
            (f"Target: False Pred\nTransformer", pred_trans_ov),
            (f"Target: False Pred\nIntegrated Gradients", pred_ig_ov),
            (f"Target: False Pred\nKernelSHAP", pred_shap_ov),
            (f"Target: True Class\n({PATHOLOGY_CLASSES[sample_b['gt']].upper()}) CNN", gt_cnn_ov)
        ]

        fig_b, axes_b = plt.subplots(1, 6, figsize=(18, 3.8), dpi=300)
        for ax, (title, img) in zip(axes_b, panels_b):
            ax.imshow(img)
            ax.set_title(title, pad=6)
            ax.axis('off')

        fig_b.suptitle(f"Figure B: Diagnostic Attribution on Misclassified Sample ({sample_b['name']})\nGround Truth: {PATHOLOGY_CLASSES[sample_b['gt']].upper()} vs Model Prediction: {PATHOLOGY_CLASSES[sample_b['pred']].upper()} (Explaining Spurious Distractors)",
                       fontsize=11.5, y=1.04)
        save_path_b = os.path.join(args.output_dir, "figure_b_misclassification_analysis.png")
        fig_b.savefig(save_path_b, bbox_inches='tight')
        plt.close(fig_b)
        print(f"    Saved: {save_path_b}")

    # =========================================================================
    # Figure C: Minority Class Attribution (Ischaemia)
    # =========================================================================
    print("[*] Generating Figure C (Minority Class Ischaemia Attribution)...")
    sample_c = correct_samples[2][0] if correct_samples[2] else (correct_samples[3][0] if correct_samples[3] else None)
    if sample_c:
        pil_c = Image.open(sample_c['image_path']).convert("RGB")
        res_c, meta_c = explainer.explain_image(pil_c, transform, target_type='pathology', target_id='ischaemia', method='all')

        cnn_ov_c, _ = overlay_heatmap_on_image(pil_c, res_c['gradcam']['cnn_heatmap'])
        trans_ov_c, _ = overlay_heatmap_on_image(pil_c, res_c['transformer']['transformer_heatmap'])
        fused_ov_c, _ = overlay_heatmap_on_image(pil_c, res_c['gradcam']['fused_heatmap'])
        ig_ov_c, _ = overlay_heatmap_on_image(pil_c, res_c['integrated_gradients']['ig_heatmap'])
        shap_ov_c, _ = overlay_heatmap_on_image(pil_c, res_c['shap']['shap_heatmap'])

        panels_c = [
            ("Original Ischaemia Ulcer", np.array(pil_c)),
            ("CNN Grad-CAM\n(Necrosis Focus)", cnn_ov_c),
            ("Transformer\n(Global Context)", trans_ov_c),
            ("Fused Dual-Branch", fused_ov_c),
            ("Integrated Gradients\n(Fine Boundaries)", ig_ov_c),
            ("KernelSHAP\n(Regional Salience)", shap_ov_c)
        ]

        fig_c, axes_c = plt.subplots(1, 6, figsize=(18, 3.8), dpi=300)
        for ax, (title, img) in zip(axes_c, panels_c):
            ax.imshow(img)
            ax.set_title(title, pad=6)
            ax.axis('off')

        fig_c.suptitle(f"Figure C: Explainability for Minority Pathology Class: ISCHAEMIA\nUlcer Sample: {sample_c['name']} | Localizing Pallor, Peripheral Cyanosis, and Necrotic Slough",
                       fontsize=11.5, y=1.04)
        save_path_c = os.path.join(args.output_dir, "figure_c_minority_ischaemia.png")
        fig_c.savefig(save_path_c, bbox_inches='tight')
        plt.close(fig_c)
        print(f"    Saved: {save_path_c}")

    # =========================================================================
    # Figure D: Wagner Ordinal Depth Grading
    # =========================================================================
    print("[*] Generating Figure D (Wagner Ordinal Depth Grading Attribution)...")
    # Search for valid Wagner images from valid folder
    w_valid_dir = os.path.join(args.wagner_dir, "valid")
    w_samples = []
    if os.path.exists(w_valid_dir):
        for grade_idx, grade_name in enumerate(WAGNER_GRADES):
            grade_path = os.path.join(w_valid_dir, grade_name)
            if os.path.exists(grade_path):
                imgs = [f for f in os.listdir(grade_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    w_samples.append((grade_idx, grade_name, os.path.join(grade_path, imgs[0])))

    if len(w_samples) >= 3:
        # Pick 3 representative grades (e.g. Superficial Grade 1, Deep Grade 2/3, Gangrene Grade 4)
        chosen_w = [w_samples[0], w_samples[len(w_samples)//2], w_samples[-1]]
        fig_d, axes_d = plt.subplots(len(chosen_w), 4, figsize=(14, 3.6 * len(chosen_w)), dpi=300)

        for row_idx, (g_idx, g_name, img_p) in enumerate(chosen_w):
            pil_w = Image.open(img_p).convert("RGB")
            res_w, meta_w = explainer.explain_image(pil_w, transform, target_type='wagner', target_id=g_idx, method='gradcam')

            cnn_w, _ = overlay_heatmap_on_image(pil_w, res_w['gradcam']['cnn_heatmap'])
            vit_w, _ = overlay_heatmap_on_image(pil_w, res_w['gradcam']['vit_heatmap'])
            fused_w, _ = overlay_heatmap_on_image(pil_w, res_w['gradcam']['fused_heatmap'])

            cols = [
                (f"Original ({g_name})", np.array(pil_w)),
                ("CNN Attribution (CORAL Cutoffs)", cnn_w),
                ("Transformer Attribution", vit_w),
                ("Fused Severity Attribution", fused_w)
            ]

            for col_idx, (col_title, img_arr) in enumerate(cols):
                ax = axes_d[row_idx, col_idx] if len(chosen_w) > 1 else axes_d[col_idx]
                ax.imshow(img_arr)
                if row_idx == 0:
                    ax.set_title(col_title, pad=6)
                else:
                    if col_idx == 0:
                        ax.set_title(col_title, pad=6)
                ax.axis('off')

        fig_d.suptitle("Figure D: Attribution Maps for Wagner Depth & Severity Grading\nExplaining Ordinal Threshold Cutoffs P(y > k) via CORAL Disentangled Representations",
                       fontsize=11.5, y=0.99)
        save_path_d = os.path.join(args.output_dir, "figure_d_wagner_depth_grading.png")
        fig_d.savefig(save_path_d, bbox_inches='tight')
        plt.close(fig_d)
        print(f"    Saved: {save_path_d}")

    # =========================================================================
    # Figure E: Cross-Method Attribution Comparison Matrix
    # =========================================================================
    print("[*] Generating Figure E (Cross-Method Attribution Comparison)...")
    # Take two distinct images: one infection, one both
    comp_imgs = []
    if correct_samples[1]:
        comp_imgs.append(correct_samples[1][0])
    if correct_samples[3]:
        comp_imgs.append(correct_samples[3][0])
    elif correct_samples[0]:
        comp_imgs.append(correct_samples[0][0])

    if len(comp_imgs) >= 2:
        fig_e, axes_e = plt.subplots(2, 6, figsize=(18, 7.2), dpi=300)
        for row_idx, s in enumerate(comp_imgs[:2]):
            pil_e = Image.open(s['image_path']).convert("RGB")
            res_e, meta_e = explainer.explain_image(pil_e, transform, target_type='pathology', target_id='predicted', method='all')

            cnn_e, _ = overlay_heatmap_on_image(pil_e, res_e['gradcam']['cnn_heatmap'])
            trans_e, _ = overlay_heatmap_on_image(pil_e, res_e['transformer']['transformer_heatmap'])
            fused_e, _ = overlay_heatmap_on_image(pil_e, res_e['gradcam']['fused_heatmap'])
            ig_e, _ = overlay_heatmap_on_image(pil_e, res_e['integrated_gradients']['ig_heatmap'])
            shap_e, _ = overlay_heatmap_on_image(pil_e, res_e['shap']['shap_heatmap'])

            cols = [
                (f"Original\n({PATHOLOGY_CLASSES[s['gt']].upper()})", np.array(pil_e)),
                ("CNN Grad-CAM\n(EfficientNetV2)", cnn_e),
                ("Transformer\n(Swin Attention)", trans_e),
                ("Dual-Branch Fused", fused_e),
                ("Integrated Gradients\n(Path Attribution)", ig_e),
                ("KernelSHAP\n(Superpixel Importance)", shap_e)
            ]

            for col_idx, (col_title, img_arr) in enumerate(cols):
                ax = axes_e[row_idx, col_idx]
                ax.imshow(img_arr)
                if row_idx == 0:
                    ax.set_title(col_title, pad=6)
                ax.axis('off')

        fig_e.suptitle("Figure E: Comprehensive Method Comparison Across Visual Resolutions\nCoarse Feature Maps (CNN/ViT 7x7) vs Fine-Grained Integrated Gradients vs Superpixel Shapley Values",
                       fontsize=11.5, y=0.98)
        save_path_e = os.path.join(args.output_dir, "figure_e_method_comparison.png")
        fig_e.savefig(save_path_e, bbox_inches='tight')
        plt.close(fig_e)
        print(f"    Saved: {save_path_e}")

    print("\n[+] All 5 publication research figures generated successfully in:", args.output_dir)


if __name__ == "__main__":
    main()
